/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/math/cos/Traits.hpp>   // Cos

//#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_floating_point
#include <math_functions.hpp>           // ::cos

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library cos.
        //#############################################################################
        class CosCudaBuiltIn
        {
        public:
            using CosBase = CosCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library cos trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Cos<
                CosCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto cos(
                    CosCudaBuiltIn const & /*cos*/,
                    TArg const & arg)
                -> decltype(::cos(arg))
                {
                    //boost::ignore_unused(cos);
                    return ::cos(arg);
                }
            };
        }
    }
}
