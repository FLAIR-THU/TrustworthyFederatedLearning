# clean up
rm -rf build/*

# build
./script/build.sh

# test for c++
cd build
ctest -V
cd ..

# test for python
python3 -m pytest test/.
