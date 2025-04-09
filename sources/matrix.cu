#include "gpu-math/matrix.hxx"

namespace gpu_math {

__global__ void fill(float data[], uint32_t size, float value);
__global__ void multiply(float source[], float target[], float destination[],
		uint16_t source_height, uint16_t source_width, uint16_t target_widht);

matrix::matrix(uint16_t height, uint16_t width) noexcept 
		: _height(height), _width(width),
		_size(width*height) {
	_host_data = std::shared_ptr<float[]>(new float[_size]);
	_bytes_count = _size*sizeof(float);
	float *device_data;
	cudaMalloc(&device_data, _bytes_count);
	_device_data = std::shared_ptr<float[]>(device_data,
			[](auto data) { cudaFree(data); });
	_on_device = true;
}

void matrix::fill(float value) noexcept
{
	gpu_math::fill<<<size(), 1>>>(device_data(), size(), value);
}

matrix matrix::multiply(const matrix &other) const noexcept {
	if (width() == other.height()) {
		matrix result(height(), other.width(), 0.f);
		gpu_math::multiply<<<{result.height(), result.width()}, 1>>>(
				device_data(), other.device_data(), result.device_data(),
				height(), width(), other.width());
		return result;
	}
	return matrix(1, 1, std::numeric_limits<float>::quiet_NaN());
}

std::ostream &operator<<(std::ostream &os, const matrix &m) {
	for (uint16_t i = 0; i < m.height(); i++) {
		os << m.get(i, 0);
		for (uint16_t j = 1; j < m.width(); j++) {
			os << ' ' << m.get(i, j);
		}
		os << '\n';
	}
	return os;
}

void gpu_math::matrix::to_host() const noexcept {
	if (_on_device) {
		cudaMemcpy(_host_data.get(), _device_data.get(),
				bytes_count(), cudaMemcpyDeviceToHost);
		_on_device = false;
	}
}

void gpu_math::matrix::to_device() const noexcept {
	if (!_on_device) {
		cudaMemcpy(_device_data.get(), _host_data.get(),
				bytes_count(), cudaMemcpyHostToDevice);
		_on_device = true;
	}
}

__global__ void fill(float data[], uint32_t size, float value) {
	uint32_t index = blockIdx.x;
	if (index < size) {
		data[index] = value;
	}
}

__global__ void multiply(float source[], float target[], float destination[],
		uint16_t source_height, uint16_t source_width, uint16_t target_widht) {
	if ( !(blockIdx.x < source_height && blockIdx.y < target_widht) ) {
		return;
	}
	uint32_t destination_index = blockIdx.x*target_widht + blockIdx.y;
	uint32_t source_start_index = blockIdx.x*source_width;
	uint32_t target_start_column = blockIdx.y;
	for (uint32_t i = 0; i < source_width; i++) {
		uint32_t source_index = source_start_index + i;
		uint32_t target_index = i*target_widht + target_start_column;
		destination[destination_index] +=
				source[source_index]*target[target_index];
	}
}

}
