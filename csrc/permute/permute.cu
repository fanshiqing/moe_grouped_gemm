/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "permute_kernels.h"

using torch::Tensor;

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num)
{
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    // initialize the workspace on the first run
    if (workspace.empty()) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
        Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
        Tensor sorted_row_id =
            torch::empty(max_expanded_token_num, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        temp_ptr, temp_ptr,
                                        temp_ptr, temp_ptr, max_expanded_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(sorted_indices);
        workspace.push_back(row_id);
        workspace.push_back(sorted_row_id);
        workspace.push_back(temp_storage);
    }

    int *indices_ptr = get_ptr<int>(indices);
    int *sorted_indices_ptr = get_ptr<int>(workspace[0]);
    int *row_id_ptr = get_ptr<int>(workspace[1]);
    int *sorted_row_id_ptr = get_ptr<int>(workspace[2]);

    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor permuted_output =
        torch::empty({num_tokens * num_topK, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor row_id_map = 
        torch::empty({num_tokens * num_topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(permuted_output, row_id_map, workspace);
}