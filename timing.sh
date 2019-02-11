time (bin/make_world | bin/step_world 0.1 1 > ref.txt)
time (bin/make_world | bin/he915/step_world_v1_lambda 0.1 1 > 1.txt)
time (bin/make_world | bin/he915/step_world_v2_function 0.1 1 > 2.txt)
time (bin/make_world | bin/he915/step_world_v3_opencl 0.1 1 > 3.txt)
time (bin/make_world | bin/he915/step_world_v4_double_buffered 0.1 1 > 4.txt)
time (bin/make_world | bin/he915/step_world_v5_packed_properties 0.1 1 > 5.txt)

DIFF1=$(diff ref.txt 1.txt)
DIFF2=$(diff ref.txt 2.txt)
DIFF3=$(diff ref.txt 3.txt)
DIFF4=$(diff ref.txt 4.txt)
DIFF5=$(diff ref.txt 5.txt)

if [ "$DIFF1" != "" ]
then
	echo "Test 1 failed!"
else
	echo "Test 1 passed!"
fi
if [ "$DIFF2" != "" ]
then
	echo "Test 2 failed!"
else
	echo "Test 2 passed!"
fi
if [ "$DIFF3" != "" ]
then
	echo "Test 3 failed!"
else
	echo "Test 3 passed!"
fi
if [ "$DIFF4" != "" ]
then
	echo "Test 4 failed!"
else
	echo "Test 4 passed!"
fi
if [ "$DIFF5" != "" ]
then
	echo "Test 5 failed!"
else
	echo "Test 5 passed!"
fi
