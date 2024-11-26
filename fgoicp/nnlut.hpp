#ifndef NNLUT_HPP
#define NNLUT_HPP
#include "common.hpp"

class NearestNeighborLUT
{
private:
	float definition;	// Definition of mesh size, default to 0.002: 500 * 500 * 500

public:
	NearestNeighborLUT(float definition);
	~NearestNeighborLUT();

private:
	void build();

};

#endif // NNLUT_HPP

