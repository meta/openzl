// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// The input the UI smoke test runs each OpenZL profile on. The numeric and
// raw-bytes profiles all read the bundled sample. The profiles that
// parse a format of their own fail on it, so each gets a file in that format,
// from the repository or written here.

const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');

const {SAMPLE_SOURCE_PATH} = require('../sample.cjs');

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');

/** 4096 float32s, the one tensor in the model checkpoints below. */
function tensorBytes() {
  const tensor = Buffer.alloc(4 * 4096);
  for (let index = 0; index < 4096; index++) {
    tensor.writeFloatLE(Math.sin(index), 4 * index);
  }
  return tensor;
}

/**
 * The sao profile's layout: a 28-byte header it stores as is, then 28-byte
 * star records it splits into fields -- two float64 coordinates, a
 * two-character spectral type, an int16 magnitude and two float32 motions.
 */
function saoFile() {
  const stars = 20_000;
  const file = Buffer.alloc(28 + 28 * stars);
  for (let index = 0; index < stars; index++) {
    const offset = 28 + 28 * index;
    file.writeDoubleLE(index / stars, offset);
    file.writeDoubleLE(-index / stars, offset + 8);
    file.write('A0', offset + 16, 'latin1');
    file.writeInt16LE(index % 1000, offset + 18);
    file.writeFloatLE(index * 1e-6, offset + 20);
    file.writeFloatLE(-index * 1e-6, offset + 24);
  }
  return file;
}

/** A u64 header length, the JSON header naming the tensor, then its bytes. */
function safetensorsFile() {
  const tensor = tensorBytes();
  const header = Buffer.from(JSON.stringify({weight: {dtype: 'F32', shape: [4096], data_offsets: [0, tensor.length]}}));
  const headerLength = Buffer.alloc(8);
  headerLength.writeBigUInt64LE(BigInt(header.length));
  return Buffer.concat([headerLength, header, tensor]);
}

/** A version 1.0 `.npy`: the magic and version, a padded header dict, then the tensor's bytes. */
function numpyFile() {
  const dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (4096,), }";
  const padding = (64 - ((10 + dict.length + 1) % 64)) % 64;
  const header = Buffer.from(`${dict}${' '.repeat(padding)}\n`, 'latin1');
  const prefix = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length, 0]);
  return Buffer.concat([prefix, header, tensorBytes()]);
}

/** A protobuf varint-length-delimited field. */
function protobufBytes(field, contents) {
  return Buffer.concat([Buffer.from([(field << 3) | 2, ...varint(contents.length)]), contents]);
}

function varint(value) {
  const bytes = [];
  for (; value > 0x7f; value = Math.floor(value / 128)) {
    bytes.push((value & 0x7f) | 0x80);
  }
  return [...bytes, value];
}

/**
 * An ONNX `ModelProto` whose graph holds one float32 initializer: the tensor's
 * dims and type, its name, and its bytes as `raw_data`.
 */
function onnxFile() {
  const tensor = Buffer.concat([
    Buffer.from([0x08, ...varint(4096), 0x10, 1]),
    protobufBytes(8, Buffer.from('weight')),
    protobufBytes(9, tensorBytes()),
  ]);
  const graph = Buffer.concat([protobufBytes(5, tensor), protobufBytes(2, Buffer.from('model'))]);
  return Buffer.concat([Buffer.from([0x08, 8]), protobufBytes(7, graph)]);
}

/** A protocol 2 pickle of a list of 4096 int32s. */
function pickleFile() {
  const items = Buffer.alloc(5 * 4096);
  for (let index = 0; index < 4096; index++) {
    items[5 * index] = 0x4a;
    items.writeInt32LE(3 * index, 5 * index + 1);
  }
  return Buffer.concat([Buffer.from([0x80, 0x02, 0x5d, 0x71, 0x00, 0x28]), items, Buffer.from([0x65, 0x2e])]);
}

/**
 * The archive `torch.save` writes: an uncompressed zip holding a pickle, the
 * tensor's bytes under `data/`, and a version. The profile splits the archive
 * by entry without unpickling anything, so the pickle is an empty dict.
 */
function pytorchFile() {
  const entries = [
    ['model/data.pkl', Buffer.from([0x80, 0x02, 0x7d, 0x71, 0x00, 0x2e])],
    ['model/data/0', tensorBytes()],
    ['model/version', Buffer.from('3\n')],
  ];
  const local = [];
  const central = [];
  let offset = 0;
  for (const [name, data] of entries) {
    const nameBytes = Buffer.from(name);
    const crc = zlib.crc32(data);
    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0);
    localHeader.writeUInt16LE(20, 4);
    localHeader.writeUInt32LE(crc, 14);
    localHeader.writeUInt32LE(data.length, 18);
    localHeader.writeUInt32LE(data.length, 22);
    localHeader.writeUInt16LE(nameBytes.length, 26);
    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0);
    centralHeader.writeUInt16LE(20, 4);
    centralHeader.writeUInt16LE(20, 6);
    centralHeader.writeUInt32LE(crc, 16);
    centralHeader.writeUInt32LE(data.length, 20);
    centralHeader.writeUInt32LE(data.length, 24);
    centralHeader.writeUInt16LE(nameBytes.length, 28);
    centralHeader.writeUInt32LE(offset, 42);
    local.push(localHeader, nameBytes, data);
    central.push(centralHeader, nameBytes);
    offset += localHeader.length + nameBytes.length + data.length;
  }
  const centralSize = central.reduce((size, part) => size + part.length, 0);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralSize, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, ...central, end]);
}

const REPO_INPUTS = {
  csv: path.join(path.dirname(SAMPLE_SOURCE_PATH), 'csv_samples', '0001.csv'),
  parquet: path.join(REPO_ROOT, 'cli', 'tests', 'sample_files', 'parquet', 'simple.parquet'),
};

const WRITTEN_INPUTS = {
  numpy: ['array.npy', numpyFile],
  onnx: ['model.onnx', onnxFile],
  pickle: ['list.pkl', pickleFile],
  pytorch: ['model.pt', pytorchFile],
  safetensors: ['model.safetensors', safetensorsFile],
  sao: ['stars.sao', saoFile],
};

/**
 * The path of a file `profile` can parse, writing it into `directory` if it is
 * not one the repository has.
 */
function profileInput(profile, directory) {
  if (Object.hasOwn(REPO_INPUTS, profile)) {
    return REPO_INPUTS[profile];
  }
  if (Object.hasOwn(WRITTEN_INPUTS, profile)) {
    const [name, contents] = WRITTEN_INPUTS[profile];
    const file = path.join(directory, name);
    fs.writeFileSync(file, contents());
    return file;
  }
  return SAMPLE_SOURCE_PATH;
}

module.exports = {profileInput};
