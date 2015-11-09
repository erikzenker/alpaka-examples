// STL
#include <iostream>

// Alpaka
#include <alpaka/alpaka.hpp>


template <typename T_Acc>
size_t globalThreadIdx(T_Acc const &acc){
    auto threadsExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
    auto nThreadsVec = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
    auto nThreads =
	nThreadsVec[0]
	+ nThreadsVec[1] * threadsExtent[0]
	+ nThreadsVec[2] * threadsExtent[0] * threadsExtent[1];

    return nThreads;
}


struct AllocKernel {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()( T_Acc const &acc) const {

	auto nThreads  = globalThreadIdx(acc);

	alpaka::mem::alloc::alloc<char>(acc, nThreads);
    }

};

int main() {


    // Set types 
    using Dim     = alpaka::dim::DimInt<3>;  
    using Size    = std::size_t;
    //using Extents = Size;
    //using Host    = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc     = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
    using Stream  = alpaka::stream::StreamCpuSync;
    using DevAcc  = alpaka::dev::Dev<Acc>;
    using DevHost = alpaka::dev::DevCpu;


    // Get the first device
    DevAcc  devAcc  (alpaka::dev::DevMan<Acc>::getDevByIdx(0));
    DevHost devHost (alpaka::dev::cpu::getDev());
    Stream  stream  (devAcc);


    // Init workdiv
    const alpaka::Vec<Dim, Size> blocks (static_cast<Size>(128),
					 static_cast<Size>(1),
					 static_cast<Size>(1));
    
    const alpaka::Vec<Dim, Size>  grid (static_cast<Size>(1), 
					 static_cast<Size>(1), 
					 static_cast<Size>(1)); 

    auto const workdiv(alpaka::workdiv::WorkDivMembers<Dim, Size>(grid, blocks));


    // Run kernel
    AllocKernel allocKernel;

    auto const exec (alpaka::exec::create<Acc> (workdiv,
						allocKernel));

    alpaka::stream::enqueue(stream, exec);
    
    return 0;
}
