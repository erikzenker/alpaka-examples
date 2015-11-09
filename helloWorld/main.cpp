// STL
#include <iostream>

// Alpaka
#include <alpaka/alpaka.hpp>

template <typename T_Acc>
size_t globalThreadIdx(T_Acc const &acc){
    auto threadsExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
    auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
    auto globalThreadIdx =
	threadIdx[0]
	+ threadIdx[1] * threadsExtent[0]
	+ threadIdx[2] * threadsExtent[0] * threadsExtent[1];

    return globalThreadIdx;
}


struct HelloWorldKernel {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()( T_Acc const &acc) const {

	std::cout << "[" << globalThreadIdx(acc) << "]" << " Hello World" << std::endl;
    }

};

int main() {


    // Set types 
    using Dim     = alpaka::dim::DimInt<3>;  
    using Size    = std::size_t;
    using Host    = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc     = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
    using Stream  = alpaka::stream::StreamCpuSync;
    using DevAcc  = alpaka::dev::Dev<Acc>;
    using DevHost = alpaka::dev::DevCpu;


    // Get the first device
    DevAcc  devAcc  (alpaka::dev::DevMan<Acc>::getDevByIdx(0));
    // DevHost devHost (alpaka::dev::cpu::getDev());
    DevHost devHost (alpaka::dev::DevMan<Host>::getDevByIdx(0));    
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
    HelloWorldKernel helloWorldKernel;

    auto const helloWorld (alpaka::exec::create<Acc> (workdiv,
						      helloWorldKernel));

    alpaka::stream::enqueue(stream, helloWorld);
    
    return 0;
}
