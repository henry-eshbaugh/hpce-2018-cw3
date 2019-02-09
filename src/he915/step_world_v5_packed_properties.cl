enum cell_flags_t{
	Curr_Cell_Fixed		= 0x1,
	Curr_Cell_Insulator	= 0x2,
	North_Cell_Insul	= 0x4,
	South_Cell_Insul	= 0x8,
	East_Cell_Insul		= 0x10,
	West_Cell_Insul		= 0x20
};

__kernel void kernel_xy(float outer, float inner, __global const float *world_state, __global float *buffer, __private cell_flags_t properties)
{

	uint x=get_global_id(0);
	uint y=get_global_id(1);
	uint w=get_global_size(0);

	uint index=y*w + x;

	if ((properties & Curr_Cell_Fixed) || (properties & Curr_Cell_Insulator)){
		// Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
		buffer[index]=world_state[index];
	} else {
		float contrib=inner;
		float acc=inner*world_state[index];
		
		// Cell above
		if(! (properties & North_Cell_Insul)) {
			contrib += outer;
			acc += outer * world_state[index-w];
		}
		
		// Cell below
		if(! (properties & South_Cell_Insul)) {
			contrib += outer;
			acc += outer * world_state[index+w];
		}
		
		// Cell left
		if(! (properties & West_Cell_Insul)) {
			contrib += outer;
			acc += outer * world_state[index-1];
		}
		
		// Cell right
		if(! (properties & East_Cell_Insul)) {
			contrib += outer;
			acc += outer * world_state[index+1];
		}
		
		// Scale the accumulate value by the number of places contributing to it
		float res=acc/contrib;
		// Then clamp to the range [0,1]
		res = min(1.0f, max(0.0f, res));
		buffer[index] = res;

	}
}
