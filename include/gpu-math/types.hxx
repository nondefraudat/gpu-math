#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>

namespace gm {

using size_t = uint32_t;
#ifdef GPU_MATH_DOUBLE_PRECISION
using real_t = double;
#else
using real_t = float;
#endif

class Shared {
	mutable enum class State : uint8_t {
		SYNCED, ON_HOST, ON_DEVICE
	} _state;
	size_t _size, _totalSize;
	mutable std::shared_ptr<real_t[]> _hostBuffer;
	mutable std::shared_ptr<real_t[]> _deviceData;
	
	// Constructors

	Shared(size_t size);

public:
	Shared(size_t size, real_t fillValue);
	Shared(Shared &other) = default;
	Shared(const Shared &other);

	// Destructor

	virtual ~Shared() = default;

	// Public accessors

	inline size_t size() const { return _size; } 
	inline size_t total_size() const { return _totalSize; }

	// Mutators

	void fill(real_t value);

protected:	
	// Protected accessors

	inline real_t *_device_data() const {
		_toDevice(); 
		return _deviceData.get();
	};
	inline real_t *_host_buffer() const {
		_toHost();
		return _hostBuffer.get();
	};
	
	// Exception tests

	inline void _index_test(size_t targetIndex) const {
		if (targetIndex >= size()) {
			throw std::out_of_range(
					"Vector index [" +
					std::to_string(targetIndex) +
					"] out of range [0, " +
					std::to_string(size() - 1) + "]");
		}
	}
	inline void _size_test(size_t targetSize) const {
		if (targetSize != size()) {
			throw std::invalid_argument(
					"Size mismatch: left[" +
					std::to_string(size()) + " v right[" +
					std::to_string(targetSize) + "]");
		}
	}
	
private:
	// Data targets

	void _toDevice() const;
	void _toHost() const;
};

}
