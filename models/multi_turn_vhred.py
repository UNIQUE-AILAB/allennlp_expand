"""
An experimental baseline model for textual knowledge driven (multi-turn) dialogue
"""
from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import GRUCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder, TimeDistributed, FeedForward
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import BLEU
from allennlp_expand.nn.parallel_beam_search import ParallelBeamSearch


@Model.register("multi_turn_vhred")
class MultiTurnVhred(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 document_encoder: Seq2VecEncoder,
                 utterance_encoder: Seq2VecEncoder,
                 context_encoder: Seq2SeqEncoder,
                 beam_size: int = 2,
                 max_decoding_steps: int = 50,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True) -> None:
        super(MultiTurnVhred, self).__init__(vocab)
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        self._beam_size = beam_size
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = ParallelBeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=self._beam_size)

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        self._max_decoding_steps = max_decoding_steps
        # Dense embedding of word level tokens.
        self._token_embedder = token_embedder
        # Document word level encoder.
        self._document_encoder = document_encoder
        # Dialogue word level encoder.
        self._utterance_encoder = TimeDistributed(utterance_encoder)
        # Sentence level encoder.
        self._context_encoder = context_encoder

        num_classes = self.vocab.get_vocab_size()

        document_output_dim = document_encoder.get_output_dim()
        utterance_output_dim = utterance_encoder.get_output_dim()
        context_output_dim = context_encoder.get_output_dim()
        decoder_output_dim = utterance_output_dim
        decoder_input_dim = token_embedder.get_output_dim() + document_output_dim + context_output_dim * 2

        self._mean_encoder = TimeDistributed(FeedForward(context_output_dim, 3, context_output_dim,
                                                         [torch.tanh, torch.tanh, lambda x: x], dropout=0.2))
        self._var_encoder = TimeDistributed(FeedForward(context_output_dim, 3, context_output_dim,
                                                        [torch.tanh, torch.tanh, lambda x: x], dropout=0.2))

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = TimeDistributed(GRUCell(decoder_input_dim, decoder_output_dim))

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(decoder_output_dim, num_classes)

    @overrides
    def forward(self,  # type: ignore
                document: Dict[str, torch.LongTensor],
                dialogue: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.
        For our experiment, an Instance should have a document field as background knowledge;
        and a dialogue field which is a ListField stores history dialogues.
        The model serves for experiments so we always have grounded target dialogue.

        To ensure at each decoding step we have last context vector, our model won't
        generate first dialogue sentence.
        """
        state = {}
        # shape: (batch_size, document_length, embedding_dim)
        embedded_document = self._token_embedder(document)
        # shape: (batch_size, document_length)
        document_mask = util.get_text_field_mask(document)
        # shape: (batch_size, document_output_dim)
        document_vec = self._document_encoder(embedded_document, document_mask)

        # training and validation
        # shape: (batch_size, sequence_num, sequence_length, embedding_dim)
        embedded_dialogue = self._token_embedder(dialogue)
        # shape: (batch_size, sequence_num, sequence_length)
        dialogue_mask = util.get_text_field_mask(dialogue, 1)

        batch_size, sequence_num, sequence_length = dialogue_mask.size()
        target_sentence_num = sequence_num - 1
        # shape: (batch_size, target_sentence_num, document_output_dim)
        state['document_vec'] = document_vec.unsqueeze(1).expand(-1, target_sentence_num, -1)
        # shape: (batch_size, sequence_num, utterance_output_dim)
        utterance_vec = self._utterance_encoder(embedded_dialogue, dialogue_mask)
        # shape: (batch_size, sequence_num, context_output_dim)
        context_vec = self._context_encoder(utterance_vec, dialogue_mask[:, :, 0])
        # shape: (batch_size, sequence_num, context_output_dim)
        context_means = self._mean_encoder(context_vec)
        context_vars = self._var_encoder(context_vec)

        # shape: (batch_size, target_sentence_num, utterance_output_dim)
        decoder_hidden = utterance_vec[:, :-1, :].contiguous()
        state['decoder_hidden'] = decoder_hidden
        # shape: (batch_size, target_sentence_num, context_output_dim)
        context_vec = context_vec[:, :-1, :].contiguous()
        state['context_vec'] = context_vec

        if self.training:
            prior_means = context_means[:, :-1, :].contiguous()
            prior_vars = context_vars[:, :-1, :].contiguous()
            post_means = context_means[:, 1:, :].contiguous()
            post_vars = context_vars[:, 1:, :].contiguous()
            state['latent_context'] = self._sampling(post_means, post_vars)
            output_dict = self._forward_loop(state, dialogue, dialogue_mask[:, 1:, :])
            output_dict['loss'] += self._get_kl_loss(prior_means, prior_vars, post_means, post_vars)
        else:
            prior_means = context_means[:, :-1, :].contiguous()
            prior_vars = context_vars[:, :-1, :].contiguous()
            # shape: (batch_size, target_sentence_num, context_output_dim)
            state['latent_context'] = self._sampling(prior_means, prior_vars)
            # Compute loss while validation and test
            output_dict = self._forward_loop(state, dialogue, dialogue_mask[:, 1:, :])

            # shape: (batch_size, target_sentence_num, utterance_output_dim)
            state['decoder_hidden'] = decoder_hidden
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if self._bleu:
                # shape: (batch_size * target_sentence_num, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"].view(batch_size * target_sentence_num, self._beam_size,
                                                                    -1)
                # shape: (batch_size * target_sentence_num, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, dialogue["tokens"][:, 1:, :].reshape(batch_size * target_sentence_num, -1))

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        # shape: (batch_size, sequence_num, beam_size, num_decoding_steps)
        predicted_indices = output_dict["predictions"]
        # print(predicted_indices.size())
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        # print(predicted_indices.shape)
        all_predicted_tokens = []
        for instance_indices in predicted_indices:
            instance_predicted_tokens = []
            for indices_sequence in instance_indices:
                if len(indices_sequence.shape) > 1:
                    # Beam search gives us the top k results for each source sentence in the batch
                    # but we just want the single best.
                    indices_sequence = indices_sequence[0]
                indices_sequence = list(indices_sequence)
                # Collect indices till the first end_symbol
                if self._end_index in indices_sequence:
                    indices_sequence = indices_sequence[:indices_sequence.index(self._end_index)]
                predicted_tokens = [self.vocab.get_token_from_index(x)
                                    for x in indices_sequence]
                instance_predicted_tokens.append(predicted_tokens)
                # print(predicted_tokens)
        all_predicted_tokens.append(instance_predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      dialogue: Dict[str, torch.LongTensor],
                      target_mask: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, sequence_num, sequence_length)
        dialogue_tokens = dialogue["tokens"]
        batch_size, sequence_num, sequence_length = dialogue_tokens.size()
        target_sentence_num = sequence_num - 1
        num_decoding_steps = sequence_length - 1
        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        # shape: (batch_size, target_sentence_num, sequence_length)
        targets = dialogue_tokens[:, 1:, :]
        # shape: (batch_size, target_sentence_num, sequence_length)
        last_predictions = dialogue_tokens.new_full((batch_size, target_sentence_num), fill_value=self._start_index)
        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size, target_sentence_num)
                input_choices = last_predictions
            else:
                # shape: (batch_size, target_sentence_num)
                input_choices = targets[:, :, timestep]
            # shape: (batch_size, target_sentence_num, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)
            # list of tensors, shape: (batch_size, target_sentence_num, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(2))
            # shape: (batch_size, target_sentence_num, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)
            # shape (predicted_classes): (batch_size, target_sentence_num)
            _, predicted_classes = torch.max(class_probabilities, -1)
            # shape (predicted_classes): (batch_size, target_sentence_num)
            last_predictions = predicted_classes
            # list of tensors, shape: (batch_size, target_sentence_num, 1)
            step_predictions.append(last_predictions.unsqueeze(-1))

        # shape: (batch_size, target_sentence_num, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)
        output_dict = {"predictions": predictions}
        # shape: (batch_size, target_sentence_num, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        # Compute loss.
        # shape: (batch_size, target_sentence_num, sequence_length)
        loss = self._get_loss(logits, targets, target_mask)
        output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        decoder_hidden = state['decoder_hidden']
        batch_size, target_sentence_num, _ = decoder_hidden.size()
        start_predictions = decoder_hidden.new_full((batch_size, target_sentence_num), fill_value=self._start_index,
                                                    dtype=torch.long)
        # shape (all_top_k_predictions): (batch_size, target_sentence_num, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, target_sentence_num, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step)
        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.
        """
        batch_size, target_sentence_num, _ = state['decoder_hidden'].size()
        # shape: (batch_size, target_sentence_num, num_classes)
        output_projections, state = \
            self._prepare_output_projections(last_predictions, state)
        # shape: (batch_size, target_sentence_num, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        context_vec = state['context_vec']
        batch_size, target_sentence_num, _ = context_vec.size()
        # shape: (batch_size, target_sentence_num, embedding_dim + document_output_dim + context_output_dim * 2)
        embedded_input = torch.cat([self._token_embedder({'tokens': last_predictions}),
                                    state['document_vec'],
                                    state['latent_context'],
                                    context_vec], dim=-1)
        # shape: (batch_size, target_sentence_num, decoder_output_dim)
        decoder_hidden = state['decoder_hidden']
        # shape: (batch_size, target_sentence_num, embedding_dim)
        decoder_hidden = self._decoder_cell(embedded_input, decoder_hidden)
        state['decoder_hidden'] = decoder_hidden
        # shape: (batch_size, target_sentence_num, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        batch_size, target_sentence_num, num_decoding_steps, _ = logits.size()
        # shape: (batch_size * target_sentence_num, num_decoding_steps)
        relevant_targets = targets[:, :, 1:].reshape(batch_size * target_sentence_num, -1)
        # shape: (batch_size * target_sentence_num, num_decoding_steps)
        relevant_mask = target_mask[:, :, 1:].reshape(batch_size * target_sentence_num, -1)
        logits = logits.view(batch_size * target_sentence_num, num_decoding_steps, -1)
        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

    def _sampling(self, means, vars):
        latent_context = torch.randn_like(means)
        latent_context = means + latent_context * torch.sqrt(torch.exp(vars))
        return latent_context

    def _get_kl_loss(self, prior_means, prior_vars, post_means, post_vars):
        batch_size = prior_means.size(0)
        kl_loss = 0.5 * (prior_vars - post_vars +
                         ((post_means - prior_means)**2 + torch.exp(post_vars)) / torch.exp(prior_vars))
        kl_loss = torch.mean(kl_loss) / batch_size
        return kl_loss
