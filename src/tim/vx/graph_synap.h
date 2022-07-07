#ifndef TIM_VX_GRAPH_SYNAP_H_
#define TIM_VX_GRAPH_SYNAP_H_

#include "graph_private.h"
//#include "metadata.hpp"
#include "synap/network.hpp"

using namespace synaptics;
using namespace synap;

namespace tim {
namespace vx {

    class GraphSynap : public GraphImpl {
        public:
        GraphSynap(ContextImpl* context, const CompileOption& options = CompileOption::DefaultOptions);
        ~GraphSynap();

        //bool CompileToBinary(void* buf, size_t* size) override;
        bool Compile() override;
        bool Run() override;

        private:
        synaptics::synap::Network _network;
        uint8_t* ebg_buffer{};
        uint8_t* nbg_buffer{};
    };
}  // namespace vx
}  // namespace tim
#endif