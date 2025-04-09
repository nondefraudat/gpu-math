#pragma once

#include <cstdint>
#include <memory>

namespace gpu_math {

class wandering_data {
	uint16_t _height, _width;
	uint32_t _size;
	size_t _bytes_count;
	mutable std::shared_ptr<float[]> _device_data;
	mutable std::shared_ptr<float[]> _host_data;
	mutable bool _on_device;

public:
	wandering_data() noexcept
		: _height(0), _width(0), _size(0), _bytes_count(0),
		_device_data(nullptr), _host_data(nullptr), _on_device(false) { }
	wandering_data(uint16_t height, uint16_t width) noexcept;

	inline uint16_t height() const noexcept { return _height; }
	inline uint16_t width() const noexcept { return _width; }

	inline uint32_t size() const noexcept { return _size; }
	inline size_t bytes_count() const noexcept { return _bytes_count; }
	
	inline float *host_data() const noexcept {
		to_host(); return _host_data.get();
	}
	inline float *device_data() const noexcept {
		to_device(); return _device_data.get();
	}

private:
	void to_host() const noexcept;
	void to_device() const noexcept;
};

}
