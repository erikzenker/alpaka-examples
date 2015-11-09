// CLIB
#include <assert.h>

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

template <typename T_Acc>
size_t globalThreadExtent(T_Acc const &acc){
    auto threadsExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
    
    auto globalThreadExtent = threadsExtent[0] * threadsExtent[0] * threadsExtent[2];

    return globalThreadExtent;
}



struct PrintBufferKernel {
    template <typename T_Acc,
	      typename T_Data,
	      typename T_Extent>
    ALPAKA_FN_ACC void operator()( T_Acc const &acc,
				   T_Data const &buffer,
				   T_Extent const extents) const {

	for(size_t i = globalThreadIdx(acc); i < extents.prod(); i += globalThreadExtent(acc)){
	    std::cout << buffer[i] << " ";
	    
	}

    }

};

struct TestBufferKernel {
    template <typename T_Acc,
	      typename T_Data,
	      typename T_Extent>
    ALPAKA_FN_ACC void operator()( T_Acc const &acc,
				   T_Data const &buffer,
				   T_Extent const extents) const {

	for(size_t i = globalThreadIdx(acc); i < extents.prod(); i += globalThreadExtent(acc)){
	    assert(buffer[i] == i);
	    
	}

    }

};


struct InitBufferKernel {
    template <typename T_Acc,
	      typename T_Data,
	      typename T_Extent,
	      typename T_Init>
    ALPAKA_FN_ACC void operator()( T_Acc const &acc,
				   T_Data &buffer,
				   T_Extent const extents,
				   T_Init initValue) const {

	for(size_t i = globalThreadIdx(acc); i < extents.prod(); i += globalThreadExtent(acc)){
	    buffer[i] = initValue;
	    
	}

    }

};


int main() {


    /***************************************************************************
     * Configure types
     **************************************************************************/
    using Dim     = alpaka::dim::DimInt<3>;
    using DimMem  = alpaka::dim::DimInt<3>;      
    using Size    = std::size_t;
    using Extents = Size;
    using Acc     = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
    using Stream  = alpaka::stream::StreamCpuSync;
    using DevAcc  = alpaka::dev::Dev<Acc>;
    using DevHost = alpaka::dev::DevCpu;


    /***************************************************************************
     * Get the first device
     **************************************************************************/
    DevAcc  devAcc  (alpaka::dev::DevMan<Acc>::getDevByIdx(0));
    DevHost devHost (alpaka::dev::cpu::getDev());
    Stream  stream  (devAcc);


    /***************************************************************************
     * Init workdiv
     **************************************************************************/
    const alpaka::Vec<Dim, Size> blocks (static_cast<Size>(128),
					 static_cast<Size>(1),
					 static_cast<Size>(1));
    
    const alpaka::Vec<Dim, Size>  grid (static_cast<Size>(1), 
					static_cast<Size>(1), 
					static_cast<Size>(1));

    auto const workdiv(alpaka::workdiv::WorkDivMembers<Dim, Size>(grid, blocks));
    
    
    /***************************************************************************
     * Create host and device buffers
     **************************************************************************/
    std::cout << "Create Buffer" << std::endl;    
    using Data = unsigned;
    const Extents nElements = 1000;

    const alpaka::Vec<DimMem, Size> extents(static_cast<Size>(nElements),
    					    static_cast<Size>(nElements),
    					    static_cast<Size>(nElements));

    alpaka::mem::buf::Buf<DevHost, Data, DimMem, Size> hostBuffer   ( alpaka::mem::buf::alloc<Data, Size>(devHost, extents));
    alpaka::mem::buf::Buf<DevAcc, Data, DimMem, Size>  deviceBuffer ( alpaka::mem::buf::alloc<Data, Size>(devAcc,  extents));


    /***************************************************************************
     * Init host buffer
     **************************************************************************/
    InitBufferKernel initBufferKernel;
    Data initValue = 0;

    auto const init (alpaka::exec::create<Acc> (workdiv,
    						initBufferKernel,
    						alpaka::mem::view::getPtrNative(deviceBuffer),
    						extents,
						initValue));
    std::cout << "Init device buffer" << std::endl;    
    alpaka::stream::enqueue(stream, init);    

    // /***************************************************************************
    //  * Write some data to host buffer
    //  **************************************************************************/
    // std::cout << "Write data to host buffer" << std::endl;        
    // for(size_t i = 0; i < extents.prod(); ++i){
    // 	alpaka::mem::view::getPtrNative(hostBuffer)[i] = i;
    // }
    

    // /***************************************************************************
    //  * Copy host to device Buffer
    //  **************************************************************************/
    // std::cout << "Copy host to device buffer" << std::endl;
    // alpaka::mem::view::copy(stream, deviceBuffer, hostBuffer, extents);    



    // /***************************************************************************
    //  * Test device Buffer
    //  **************************************************************************/
    // TestBufferKernel testBufferKernel;
    // auto const test (alpaka::exec::create<Acc> (workdiv,
    // 						testBufferKernel,
    // 						alpaka::mem::view::getPtrNative(deviceBuffer),
    // 						extents));


    // std::cout << "Test device buffer" << std::endl;        
    // alpaka::stream::enqueue(stream, test);


    /***************************************************************************
     * Create view
     **************************************************************************/
    using DataView = alpaka::mem::view::View<DevHost,
					     Data,
					     alpaka::dim::DimInt<1>,
					     Size>;

    
    auto view = alpaka::mem::view::createView<DataView>(hostBuffer);


    return 0;
    
}
