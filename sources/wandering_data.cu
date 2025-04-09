#include "wandering_data.hxx"

namespace gpu_math {

wandering_data::wandering_data(uint16_t height, uint16_t width) noexcept
		: _height(height), _width(width) {
	_size = static_cast<uint32_t>(_height)*_width;
	_host_data = std::shared_ptr<float[]>(new float[_size]);
	_bytes_count = static_cast<size_t>(_size)*sizeof(float);
	float *device_data;
	cudaMalloc(&device_data, _bytes_count);
	_device_data = std::shared_ptr<float[]>(device_data,
			[](auto data) { cudaFree(data); });
	_on_device = true;
}

void wandering_data::to_host() const noexcept {
	if (_on_device) {
		cudaMemcpy(_host_data.get(), _device_data.get(),
				bytes_count(), cudaMemcpyDeviceToHost);
		_on_device = false;
	}
}

void wandering_data::to_device() const noexcept {
	if (!_on_device) {
		cudaMemcpy(_device_data.get(), _host_data.get(),
				bytes_count(), cudaMemcpyHostToDevice);
		_on_device = true;
	}
}

}
