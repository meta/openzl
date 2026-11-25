## Installation

From PyPI:

```
pip install openzl
```

From source:

```
git clone https://github.com/facebook/openzl
cd openzl/py
pip install .
```

## Running the example

This example is self-contained and can be run after installing OpenZL and NumPy.
The full source can be found in `examples/py/quick_start.py`.

```
python3 examples/py/quick_start.py
```

## Imports

```python
--8<-- "src/examples/py/quick_start.py:imports"
```

## Generate test data

In this example, we'll work with very simple test data that is little-endian int64 data that is either low cardinality or sorted.

```python
--8<-- "src/examples/py/quick_start.py:generate_test_data"
```

## Setting up a simple Compressor

The [Compressor][openzl.ext.Compressor] tells OpenZL how to compress the data it receives.
This is how OpenZL is specialized to build a format-specific compressor for a particular use case.
For now, we will just tell OpenZL to use the generic compression backend [graphs.Compress][openzl.ext.graphs.Compress].
Later on, after we get through the basics, we'll build more complex compressors.

```python
--8<-- "src/examples/py/quick_start.py:simple_compressor"
```

## Bytes compression & decompression

OpenZL's compression interface can accept more than just a single input of bytes, but for now we will keep it simple.
The compression method needs to take in the `compressor` we've built, but the decompression doesn't, because we write the steps needed to decompress the data into the compressed representation.

```python
--8<-- "src/examples/py/quick_start.py:compress_bytes"
```

## Putting it all together

We can now build a function that tells us the compressed size when using the simple [Compressor][openzl.ext.Compressor] that we've built. The same testing function can be built for all future compressors.

```python
--8<-- "src/examples/py/quick_start.py:test_simple_compressor"
```

```
Simple bytes compressor: 4.14
```

## Building a simple Int64 compressor

So far, we haven't told OpenZL anything about the data we're compressing.
All it knows is that it is compressing bytes.
However, OpenZL excels when it knows the format of the data it is compressing.

The simplest case is just telling OpenZL that the data is numeric data with a certain width. In this case, we'll compress little-endian int64 data.
The only difference from `build_simple_compressor()` is that we added a node that converts from serial data to numeric data.
The [Compress graph][openzl.ext.graphs.Compress] accepts any input type and handles it appropriately.

```python
--8<-- "src/examples/py/quick_start.py:simple_int64_compressor"
```

```
Simple Int64 compressor: 5.18
```

## Expanding on the Int64 compressor

Just telling OpenZL the data is numeric will bring serious improvements.
However, if you know more about your data, you can do even better.

```python
--8<-- "src/examples/py/quick_start.py:better_int64_compressor"
```

```
Better Int64 compressor: 5.26
```

## Compressing native-endian numeric data

OpenZL can also compress native-endian numeric data from a NumPy array, PyTorch tensor, a dlpack tensor, or any object which implements the Buffer Protocol.
The compressor you build is exactly the same, it just starts with numeric data instead of serial data.

This example shows that OpenZL correctly interprets the NumPy array as uint32 values, since it is able to compress the data with the constant graph.


```python
--8<-- "src/examples/py/quick_start.py:compress_numpy_array"
```
