/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "groupedgemm_varM_template.h"
#include "groupedgemm_varK_template.h"

template class MoeGemmRunner<float, float, half>;