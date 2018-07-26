import tensorflow as tf
import tensorflow.contrib.layers as layers

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths=None,
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            scope=scope))
        outputs = tf.concat((fw_outputs, bw_outputs), 2)    # foreach i: outputs[i] = [fw_outputs[i], bw_outputs[i]]

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1,
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                  isinstance(bw_state, tuple) and
                  len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state
            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)
        return outputs, state


def word_attention(inputs,
                   aspect_size,
                   initializer=layers.xavier_initializer(),
                   activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context matrix (constant within task of interest).

    Args:
        inputs:[document_size*sentence_size, word_size, word_encode_size]

    Returns:
        outputs: Tensor of shape [document_size*sentence_size, aspect_size, word_encode_size]
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    r = aspect_size
    vector_size = inputs.get_shape()[-1].value    # same as word_encode_size
    with tf.variable_scope(scope or 'attention') as scope:
        W_s1 = tf.get_variable(name='W_s1',
                               shape=[vector_size, vector_size],
                               initializer=initializer,
                               dtype=tf.float32)

        W_s2 = tf.get_variable(name='W_s2',
                               shape=[r, vector_size],
                               initializer=initializer,
                               dtype=tf.float32)

        # [batch_size, n, 2u]
        H = inputs
        batch_size = tf.shape(H)[0]

        # [batch_size, r, n]
        A = tf.nn.softmax(
            tf.map_fn(
                lambda x: tf.matmul(W_s2, x),
                tf.tanh(
                    tf.map_fn(
                        lambda x: tf.matmul(W_s1, tf.transpose(x)),
                        H
                    )
                )
            )
        )

        # [batch_size, r, 2u]
        M = tf.matmul(A, H)

        A_T = tf.transpose(A, perm=[0, 2, 1])
        tile_eye = tf.tile(tf.eye(r), [batch_size, 1])
        tile_eye = tf.reshape(tile_eye, [-1, r, r])
        AA_T = tf.matmul(A, A_T) - tile_eye
        P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

        # penalizations_batch_avg = tf.reduce_mean(tf.clip_by_value(P, 1e-8, 3))  # avoid nan
        # penalizations_batch_avg = tf.reduce_mean(P + 1e-6)  # avoid nan

        return M, P


def sentence_attention(inputs,
                       initializer=layers.xavier_initializer(),
                       activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context matrix (constant within task of interest).

    Args:
        inputs: [document_size, sentence_size, aspect_size, sentence_encode_size]

    Returns:
        outputs: Tensor of shape [document_size, aspect_size, sentence_encode_size]
    """
    assert len(inputs.get_shape()) == 4 and inputs.get_shape()[-1].value is not None

    (document_size, sentence_size, aspect_size, encode_size) = tf.unstack(tf.shape(inputs))
    context_vector_size = inputs.get_shape()[-1].value  # same as sentence_encode_size
    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_matrix',
                                                   shape=[context_vector_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)

        # project inputs.input_size to output_size with a fully_connected layer
        #   from shape([document_size, senetence_size, aspect_size, sentence_vector_size]) to shape([document_size, senetence_size, aspect_size, context_vector_size])
        inputs_projection = layers.fully_connected(inputs, context_vector_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        # shape(attention_weights_matrix) : [document_size, senetence_size, aspect_size]
        inputs_projection_deranked_for_matmul = tf.reshape(inputs_projection, shape=[document_size * sentence_size * aspect_size, context_vector_size])
        attention_weights_matrix = tf.nn.softmax(
            tf.reshape(
                tf.matmul(inputs_projection_deranked_for_matmul, tf.reshape(attention_context_vector, shape=[context_vector_size,1])),
                shape=[document_size, sentence_size, aspect_size]
            ),
            dim=1
        )

        # shape(attention_weights_matrix) : [document_size, senetence_size, aspect_size, 1]
        attention_weights_matrix_for_multiply = tf.reshape(attention_weights_matrix, [document_size, sentence_size, aspect_size, 1])

        # shape(outputs_by_aspects) : [document_size, aspect_size, sentence_encode_size]
        outputs_by_aspects = tf.reduce_sum(tf.multiply(inputs, attention_weights_matrix_for_multiply), axis=1)

        # # shape(outputs) : [document_size, sentence_encode_size]
        # outputs = tf.reduce_mean(outputs_by_aspects, axis=1)

        return outputs_by_aspects


def aspect_attention(inputs,
                       initializer=layers.xavier_initializer(),
                       activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context matrix (constant within task of interest).

    Args:
        inputs: [document_size, aspect_size, aspect_encode_size]

    Returns:
        outputs: Tensor of shape [document_size, aspect_encode_size]
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    (document_size, aspect_size, encode_size) = tf.unstack(tf.shape(inputs))
    context_vector_size = inputs.get_shape()[-1].value  # same as aspect_encode_size
    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[context_vector_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)

        # project inputs.input_size to output_size with a fully_connected layer
        #   from shape([document_size, aspect_size, aspect_encode_size]) to shape([document_size, aspect_size, context_vector_size])
        inputs_projection = layers.fully_connected(inputs, context_vector_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        # shape(attention_weights_matrix) : [document_size, aspect_size]
        attention_weights_matrix = tf.nn.softmax(
            tf.reshape(
                tf.matmul(
                    tf.reshape(inputs_projection, shape=[document_size * aspect_size, context_vector_size]),
                    tf.reshape(attention_context_vector, shape=[context_vector_size, 1])
                ),
                shape=[document_size, aspect_size]
            ),
            dim=1
        )

        # shape(attention_weights_matrix_for_multiply) : [document_size, aspect_size, 1]
        attention_weights_matrix_for_multiply = tf.reshape(
            attention_weights_matrix,
            shape=[document_size, aspect_size, 1]
        )

        outputs = tf.reduce_sum(
            tf.multiply(
                attention_weights_matrix_for_multiply,
                inputs
            ),
            axis=1
        )

        return outputs



