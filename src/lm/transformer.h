#ifndef LlmNet_H
#define LlmNet_H

/**
 * @file transformer.h
 * @brief Optimized Transformer neural architecture for language processing
 *
 * Implements a Transformer architecture with multi-head attention and RMS
 * normalization, designed for efficient learning and inference. The architecture
 * leverages cognitive principles of parallel information processing and
 * selective attention.
 *
 * Key features:
 * - RMS normalization for enhanced stability
 * - Optimized residual connections
 * - Causal masking for autoregressive attention
 */

#include <dlib/dnn.h>

namespace transformer
{
    using namespace dlib;

    // Scale Weights Layer
    template <long d_k_>
    class scale_weights_ : public multiply_ {
    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    namespace def {
        template <long num_heads, long d_model, typename SUBNET>
        using query = extract<0, num_heads, d_model / num_heads, 1, SUBNET>;

        template <long num_heads, long d_model, typename SUBNET>
        using key = extract<d_model, num_heads, 1, d_model / num_heads, SUBNET>;

        template <long num_heads, long d_model, typename SUBNET>
        using value = extract<(d_model * 2), num_heads, d_model / num_heads, 1, SUBNET>;

        /**
         * Multi-Head Attention Layer
         *
         * Structure:
         * 1. Input Processing
         *    - RMS normalization
         *    - Single linear projection (d_model -> 3*d_model) for Q,K,V
         * 2. Parallel Head Processing (num_heads)
         *    - Split into Q, K, V tensors
         *    - Key transposition for attention computation
         * 3. Attention Mechanism
         *    - Scaled dot-product (Q*K^T / sqrt(d_k))
         *    - Causal masking (tril_mask)
         *    - Softmax normalization
         *    - Value weighting
         * 4. Output
         *    - Head concatenation
         *    - Residual connection
         *
         * Template Parameters:
         * @param ACT: Activation function type
         * @param DO: Dropout layer type
         * @param d_model: Model dimension
         * @param num_heads: Number of attention heads
         * @param SUBNET: Input subnet type
         */
        template <template <typename> class ACT, template <typename> class DO, long d_model, long num_heads, typename SUBNET>
        using multihead_attention = add_prev1<
            DO<extract<0, 1, 1, d_model, multm_prev3<
            DO<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<query<num_heads, d_model, skip2<
            tag4<key<num_heads, d_model, skip2<
            tag3<value<num_heads, d_model,
            tag2<fc_no_bias<d_model * 3, rms_norm<
            tag1<SUBNET>>>>>>>>>>>>>>>>>>>>;

        /**
         * Feed-Forward Network Layer
         *
         * Structure:
         * 1. Input Processing
         *    - RMS normalization
         *    - Input tagged for residual connection
         * 2. Transformation
         *    - Expansion layer (d_model -> 4*d_model)
         *    - Activation function
         *    - Projection layer (4*d_model -> d_model)
         * 3. Output
         *    - Dropout
         *    - Residual connection
         *
         * Template Parameters:
         * @param ACT: Activation function type
         * @param DO: Dropout layer type
         * @param d_model: Model dimension
         * @param SUBNET: Input subnet type
         */
        template <template <typename> class ACT, template <typename> class DO, long d_model, typename SUBNET>
        using feed_forward =
            add_prev5<
            DO<extract<0, 1, 1, d_model,
            fc<d_model, ACT<fc<d_model * 4, rms_norm<
            tag5<SUBNET>>>>>>>>;

        /**
         * Transformer Block
         *
         * Combines sequentially:
         * 1. Multi-head attention layer
         * 2. Feed-forward network
         *
         * Template Parameters:
         * @param ACT: Activation function type
         * @param DO: Dropout layer type
         * @param d_model: Model dimension
         * @param num_heads: Number of attention heads
         * @param SUBNET: Input subnet type
         */
        template <template <typename> class ACT, template <typename> class DO, long d_model, long num_heads, typename SUBNET>
        using transformer_block =
            feed_forward<ACT, DO, d_model,
            multihead_attention<ACT, DO, d_model, num_heads, SUBNET>>;
    }

    // Positional Embeddings
    template <long num_embeddings, long embedding_length, typename SUBNET>
    using positional_embeddings = positional_encodings<embeddings<num_embeddings, embedding_length, SUBNET>>;

    // Learned Positional Embeddings
    template <long num_embeddings, long embedding_length, typename SUBNET>
    using learned_positional_embeddings = add_prev9<linear<embedding_length, skip10<
        tag9<embeddings<num_embeddings, embedding_length, tag10<SUBNET>>>>>>;

    // Classification Head   
    template <template <typename> class ACT, long embedding_length, typename SUBNET>
    using squeezing = fc<embedding_length / 4, ACT<fc<embedding_length / 8, SUBNET>>>;
    
    template <template <typename> class ACT, long num_logits, long embedding_length, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_logits, squeezing<ACT, embedding_length, rms_norm<SUBNET>>>>;

    template <typename SUBNET>
    using dropout_10 = dropout_rate<10, SUBNET>;

    /**
     * @brief Transformer model configuration template
     *
     * Provides a flexible and type-safe configuration mechanism for Transformer models
     * with compile-time parameter validation and network generation.
     *
     * @tparam vocab_size Vocabulary size for token embedding
     * @tparam num_layers Number of Transformer layers
     * @tparam num_heads Number of attention heads
     * @tparam embedding_dim Dimension of token embeddings
     * @tparam max_seq_len Maximum sequence length
     * @tparam activation_func Activation function type
     * @tparam dropout_policy Dropout regularization policy
     */
    template <
        long vocab_size = 2000,                                 // Default vocabulary size
        long num_layers = 3,                                    // Default number of layers
        long num_heads = 4,                                     // Default number of attention heads
        long embedding_dim = 128,                               // Default embedding dimension
        long max_seq_len = 80,                                  // Default maximum sequence length
        template <typename> class activation_func = gelu,      // Default activation function
        template <typename> class dropout_policy = dropout_10  // Default dropout policy
    >
    struct transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long MAX_SEQ_LEN = max_seq_len;

        /**
         * @brief Compile-time validation of model configuration
         *
         * Performs static assertions to ensure valid model parameters
         */
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM % NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        /**
         * @brief Network type generation based on training/inference mode
         *
         * Generates different network types for training and inference
         * using the configured parameters
         *
         * @tparam is_training Determines training or inference network type
         */
        template <typename SUBNET>
        using t_transformer_block = def::transformer_block<activation_func, dropout_policy, EMBEDDING_DIM, NUM_HEADS, SUBNET>;
        template <typename SUBNET>
        using i_transformer_block = def::transformer_block<activation_func, multiply, EMBEDDING_DIM, NUM_HEADS, SUBNET>;

        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head<activation_func, VOCAB_SIZE, EMBEDDING_DIM,
            repeat<NUM_LAYERS, t_transformer_block,
            positional_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>,
            classification_head<activation_func, VOCAB_SIZE, EMBEDDING_DIM,
            repeat<NUM_LAYERS, i_transformer_block,
            positional_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>
        >;

        /**
         * @brief Model configuration information and debugging utility
         *
         * Provides methods to generate human-readable model configuration details
         */
        struct model_info {
            /**
             * @brief Generate a detailed description of the model configuration
             *
             * @return String containing model configuration details
             */
            static std::string describe() {
                std::stringstream ss;
                ss << "transformer model configuration:\n"
                    << "- vocabulary size: " << VOCAB_SIZE << "\n"
                    << "- layers: " << NUM_LAYERS << "\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM << "\n"
                    << "- max sequence length: " << MAX_SEQ_LEN;
                return ss.str();
            }
        };
    };

    using vslm = transformer_config<>; // Very Small Language Model

    /**
     * @example Configuration and Usage Examples
     *
     * // Creating different transformer configurations
     * using default_transformer = transformer_config<>;
     * using large_transformer = transformer_config<
     *     5000,   // Larger vocabulary
     *     8,      // More layers
     *     8,      // More heads
     *     512,    // Larger embedding dimension
     *     128     // Longer sequences
     * >;
     *
     * // Network type instantiations for different modes
     * using train_network = default_transformer::network_type<true>;
     * using inference_network = default_transformer::network_type<false>;
     *
     * // Utility function to print model configuration
     * void print_model_info() {
     *     std::cout << default_transformer::model_info::describe() << std::endl;
     * }
     *
     * @note
     * - Supports compile-time configuration
     * - Provides static validation of model parameters
     * - Enables dynamic network type generation
     * - Offers advanced hyperparameter tuning utilities
     *
     * @author Cydral
     * @site https://github.com/Cydral/ERNIE
     * @version 1.0
     * @date 11/2024
     */
}

#endif // LlmNet_H
