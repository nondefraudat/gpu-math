#include "Vector.hxx"
#include <exception>
#include <string>

// Device methods declaration

__global__ void _fill(float deviceData[], uint32_t size, float value);
__global__ void _add(float source[], float target[], uint32_t size);
__global__ void _subtract(float source[], float target[], uint32_t size);

// Public methods

Vector::Vector(uint32_t size) : _size(size),
		_totalSize(size*sizeof(float)),
		_hostBuffer(new float[size]) {
	float* deviceData;
	cudaMalloc(&deviceData, _totalSize);
	_deviceData = std::shared_ptr<float[]>(
				deviceData, [](auto d) { cudaFree(d); });
	_state = State::ON_DEVICE;
}

Vector::Vector(uint32_t size, float fillValue)
		: Vector(size) { fill(fillValue); }

Vector::Vector(const Vector &vector) : Vector(vector.size()) {
	_toDevice(); vector._toDevice();
	cudaMemcpy(_deviceData.get(), vector._deviceData.get(),
			total_size(), cudaMemcpyDeviceToDevice);
}

void Vector::fill(float value) {
	_toDevice();
	_fill<<<_size, 1>>>(_deviceData.get(), size(), value);
}

void Vector::add(const Vector &vector) {
	_size_test(_size, vector._size);
	_toDevice(); vector._toDevice();
	_add<<<size(), 1>>>(_deviceData.get(),
			vector._deviceData.get(), size());
}

void Vector::subtract(const Vector &vector) {
	_size_test(size(), vector.size());
	_toDevice(); vector._toDevice();
	_subtract<<<size(), 1>>>(_deviceData.get(),
			vector._deviceData.get(), size());
}

Vector &Vector::operator=(float value) {
	fill(value); return *this;
}

Vector &Vector::operator+=(const Vector &target) {
	add(target); return *this;
}

Vector Vector::operator+(const Vector &target) const {
	_size_test(_size, target._size);
	Vector result(*this);
	result.add(target);
	return result;
}


Vector &Vector::operator-=(const Vector &target) {
	subtract(target); return *this;
}

Vector Vector::operator-(const Vector &target) const {
	_size_test(_size, target._size);
	Vector result(*this);
	result.subtract(target);
	return result;
}

float Vector::operator[](uint32_t index) const {
	if (index >= _size) {
		throw std::out_of_range(
				"Vector index [" + std::to_string(index) +
				"] out of range [0, " +
				std::to_string(_size - 1) + "]");
	}
	_toHost();
	return _hostBuffer[index];
}

std::ostream &operator<<(std::ostream &stream, const Vector &vector) {
	stream << vector[0];
	for (uint32_t i = 1; i < vector.size(); i++) {
		stream << " " << vector[i];
	}
	return stream;
}


// Private methods

void Vector::_size_test(uint32_t left, uint32_t right) {
	if (left != right) {
		throw std::invalid_argument(
				"Size mismatch: left[" +
				std::to_string(left) + " v right[" +
				std::to_string(right) + "]");
	}
}

void Vector::_toHost() const
{
	if (_state == State::ON_DEVICE) {
		cudaMemcpy(_hostBuffer.get(), _deviceData.get(),
				total_size(), cudaMemcpyDeviceToHost);
		_state = State::ON_HOST;
	}
}

void Vector::_toDevice() const {
	if (_state == State::ON_HOST) {
		cudaMemcpy(_deviceData.get(), _hostBuffer.get(),
				total_size(), cudaMemcpyHostToDevice);
		_state = State::ON_DEVICE;
	}
}

// Device method definitions

__global__ void _fill(float deviceData[], uint32_t size, float value) {
	uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size) {
		deviceData[index] = value;
	}
}

__global__ void _add(float source[], float target[], uint32_t size) {
	uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size) {
		source[index] += target[index];
	}
}

__global__ void _subtract(float source[], float target[], uint32_t size) {
	uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size) {
		source[index] -= target[index];
	}
}
