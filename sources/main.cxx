#include "Vector.hxx"
#include <iostream>

int main() {
	uint32_t size = 64;
	Vector left(size, 1.f), right(size, 2.14f);
	Vector result = left - right;
	std::cout << left << "\n-\n" << right <<
			"\n=\n" << result << std::endl;
	return 0;
}