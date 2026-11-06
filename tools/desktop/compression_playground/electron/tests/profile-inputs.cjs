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

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..', '..');

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
