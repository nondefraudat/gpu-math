#include "gpu-math/types.hxx"

namespace gm {

// Device method declarations

__global__ void _fill(float deviceData[], uint32_t size, float value);

// Constructors

Shared::Shared(size_t size) 
		: _size(size), _hostBuffer(new real_t[size]) {
	_totalSize = _size*sizeof(real_t);
	real_t *deviceData;
	cudaMalloc(&deviceData, _totalSize);
	_deviceData = std::shared_ptr<real_t[]>(
			deviceData, [](auto d) { cudaFree(d); });
	_state = State::ON_DEVICE;
}

Shared::Shared(size_t size, real_t fillValue)
		: Shared(size) { fill(fillValue); }

Shared::Shared(const Shared &other) : Shared(other.size()) {
	cudaMemcpy(_device_data(), other._device_data(),
			total_size(), cudaMemcpyDeviceToDevice);
}

// Mutators

void Shared::fill(real_t value) {
	_toDevice();
	_fill<<<_size, 1>>>(_deviceData.get(), size(), value);
}

// Data targets

void Shared::_toDevice() const {
	if (_state == State::ON_HOST) {
		cudaMemcpy(_deviceData.get(), _hostBuffer.get(),
				total_size(), cudaMemcpyHostToDevice);
		_state = State::ON_DEVICE;
	}
}

void Shared::_toHost() const {
	if (_state == State::ON_DEVICE) {
		cudaMemcpy(_hostBuffer.get(), _deviceData.get(),
				total_size(), cudaMemcpyDeviceToHost);
		_state = State::ON_HOST;
	}
}

// Device method definitions

__global__ void _fill(float deviceData[], uint32_t size, float value) {
	uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size) {
		deviceData[index] = value;
	}
}

}