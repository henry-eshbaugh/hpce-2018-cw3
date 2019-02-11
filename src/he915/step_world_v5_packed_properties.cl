__kernel void kernel_xy(float inner, float outer, __global const float *world_state, __global float *buffer, __global uchar *properties)
{

	uint x=get_global_id(0);
	uint y=get_global_id(1);
	uint w=get_global_size(0);

	uint index=y*w + x;

	__private uchar props = properties[index];

	if (props & 0x01){
		// Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
		buffer[index]=world_state[index];
	} else {
		float contrib=inner;
		float acc=inner*world_state[index];
		
		// Cell above
		if(! (props & 0x02)) {
			contrib += outer;
			acc += outer * world_state[index-w];
		}
		
		// Cell below
		if(! (props & 0x04)) {
			contrib += outer;
			acc += outer * world_state[index+w];
		}
		
		// Cell left
		if(! (props & 0x08)) {
			contrib += outer;
			acc += outer * world_state[index-1];
		}
		
		// Cell right
		if(! (props & 0x10)) {
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
