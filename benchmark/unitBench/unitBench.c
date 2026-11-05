// Copyright (c) Meta Platforms, Inc. and affiliates.

/* unitBench
 * measure speed / performance of any zstrong's transform on any input
 */

/// MinGW: Use the ANSI stdio functions (e.g. to get correct printf for 64-bits)
#undef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1

/* ===   Dependencies   === */
#include <assert.h>
#include <stdio.h>  // printf, fflush
#include <stdlib.h> // malloc, free

#include "benchmark/unitBench/benchfn.h"
#include "tools/fileio/fileio.h"

#include "benchmark/unitBench/benchList.h" // list of functions & graphs to benchmark
#include "openzl/zl_compress.h"
#include "openzl/zl_data.h"       // ZL_DataArenaType
#include "openzl/zl_decompress.h" // ZL_DCtx_create, ZL_DCtx_setStreamArena

/* Global parameters */
#define ONE_RUN_BENCH_TIME_MS 800
#define TOTAL_BENCH_TIME_MS_DEFAULT 1900

typedef struct {
    int clevel;
    int decompressOnly;
    int noDecompress;
    int genericIntParam;
    ZL_DataArenaType sat;
    int satSet; // 0: keep the library defaults
    size_t blockSize;
    unsigned total_bench_time_ms;
    int notification;
    int memory;
    bool saveArtifact;
} BenchParams;

/* Generic macros and functions */
#define EXIT(...)                     \
    {                                 \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, " \n");       \
        exit(1);                      \
    }

static void* malloc_orDie(size_t s)
{
    void* const r = malloc(s);
    if (s && r == NULL)
        EXIT("allocation error (requesting %zu bytes)", s);
    return r;
}

#if defined(__MACH__) || defined(__linux__)

#    include <sys/resource.h>
static size_t getProcessMemUsage(int children)
{
    struct rusage stats;
    if (getrusage(children ? RUSAGE_CHILDREN : RUSAGE_SELF, &stats) == 0)
        return (size_t)stats.ru_maxrss;
    return 0;
}

#else
static size_t getProcessMemUsage(int ignore)
{
    (void)ignore;
    return 0;
}
#endif

/* @return: funcID (necessarily between 0 and NB_FUNCS) if present
 *          -1 on error (fname not present) */
static int funcID(const char* fname);

static size_t sumST(const size_t* a, size_t s)
{
    size_t total = 0;
    for (size_t n = 0; n < s; n++) {
        total += a[n];
    }
    return total;
}

static BMK_runTime_t fasterOne(BMK_runTime_t rt1, BMK_runTime_t rt2)
{
    return (rt1.nanoSecPerRun < rt2.nanoSecPerRun) ? rt1 : rt2;
}

/* result display specialized for zstrong decompression */
static void zsDecompressResult(
        const char* srcname,
        const char* cgraphName,
        BMK_runTime_t rt,
        size_t cSize)
{
    double const sec           = rt.nanoSecPerRun / 1000000000.;
    double const nbRunsPerSec  = 1. / sec;
    double const nbBytesPerSec = nbRunsPerSec * (double)rt.sumOfReturn;

    printf("decode %zu KB from %s compressed with %s graph (R:x%.2f) in %.2f ms  ==> %.1f MB/s",
           rt.sumOfReturn >> 10,
           srcname,
           cgraphName,
           (double)rt.sumOfReturn / (double)cSize,
           sec * 1000.,
           nbBytesPerSec / (1 << 20));
}

static BMK_runTime_t bench_zsDecode(
        const void* cPtrs[],
        size_t cSizes[],
        size_t nbBlocks,
        size_t blocksSize,
        const void* orig,
        void* payload,
        const BenchParams* benchParams,
        const char* filename,
        const char* functionName)
{
    size_t const cSize    = sumST(cSizes, nbBlocks);
    size_t const decSize  = nbBlocks * blocksSize;
    void* const decBuffer = malloc_orDie(decSize);
    void** const decPtrs  = malloc_orDie(nbBlocks * sizeof(*decPtrs));
    size_t* const decCapacities =
            malloc_orDie(nbBlocks * sizeof(*decCapacities));

    decPtrs[0]   = decBuffer;
    char* decPtr = decBuffer;
    for (size_t bn = 0; bn < nbBlocks; bn++) {
        decPtrs[bn]       = decPtr;
        decCapacities[bn] = blocksSize;
        decPtr += blocksSize;
    }

    int const fID = funcID("zs2_decompress");
    assert(fID != -1);
    Bench_Entry const zsDec = scenarioList[fID];

    BMK_benchParams_t params = {
        .benchFn       = zsDec.func,
        .benchPayload  = payload,
        .initFn        = zsDec.init,
        .initPayload   = payload,
        .errorFn       = NULL,
        .blockCount    = nbBlocks,
        .srcBuffers    = cPtrs,
        .srcSizes      = cSizes,
        .dstBuffers    = decPtrs,
        .dstCapacities = decCapacities,
        .blockResults  = NULL,
    };

    // bench
    BMK_timedFnState_shell tfnShell; // allocate on stack
    BMK_timedFnState_t* const tfn = BMK_initStatic_timedFnState(
            &tfnShell,
            sizeof(tfnShell),
            benchParams->total_bench_time_ms,
            ONE_RUN_BENCH_TIME_MS);
    if (tfn == NULL)
        exit(1);
    BMK_runOutcome_t r;
    BMK_runTime_t best = { .nanoSecPerRun = ONE_RUN_BENCH_TIME_MS * 1000000 };
    int loopCount      = 0;
    while (!BMK_isCompleted_TimedFn(tfn)) {
        r = BMK_benchTimedFn(tfn, params);
        if (!BMK_isSuccessful_runOutcome(r)) {
            EXIT("Error : decompression failed");
        }
        best = fasterOne(best, BMK_extract_runTime(r));
        if (benchParams->notification) {
            zsDecompressResult(filename, functionName, best, cSize);
            if (benchParams->memory) {
                size_t const programBytesSelf = getProcessMemUsage(0);
                if (programBytesSelf) {
                    printf(" ==> MaxRSS=%9zu ", programBytesSelf);
                }
            }
            printf("  \r%4u - \r", ++loopCount); // notification
            fflush(NULL);
        }
    }

    if (memcmp(decBuffer, orig, BMK_extract_runTime(r).sumOfReturn)) {
        EXIT("corruption detected: regenerated data differs from original!");
    }

    free(decPtrs);
    free(decBuffer);

    return best;
}

static void BMK_displayResult_default(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize)
{
    (void)srcname;
    (void)fname;
    // default result display
    double const sec           = rt.nanoSecPerRun / 1e+9;
    double const nbRunsPerSec  = 1. / sec;
    double const nbBytesPerSec = nbRunsPerSec * (double)srcSize;

    printf("process %s (%zu KB) with %s in %.2f ms  ==> %.1f MB/s  (%zu)",
           srcname,
           srcSize >> 10,
           fname,
           sec * 1000.,
           nbBytesPerSec / (1 << 20),
           rt.sumOfReturn);
}

/* generic result format, for csv parsing
 * return source name and size, function name and result,
 * and processing time per run in us (microsecond) */
static void csvResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize)
{
    printf("%s, %zu, %s, %zu, %.1f",
           srcname,
           srcSize,
           fname,
           rt.sumOfReturn,
           (double)rt.nanoSecPerRun / 1000. /* us format */);
    fflush(NULL);
}

static void saveBuf(const char* filename, const void* buffer, size_t size)
{
    FILE* const fw  = fopen(filename, "wb");
    size_t const ws = fwrite(buffer, 1, size, fw);
    assert(ws == size);
    (void)ws;
    fclose(fw);
}

static BMK_runTime_t bench_function(
        void** dstPtrs,
        size_t* dstSizes,
        size_t nbBlocks,
        size_t dstBlockCapacity,
        const void* src,
        size_t srcSize,
        BMK_benchFn_t fn,
        BMK_initFn_t init,
        void* payload,
        BenchParams bp,
        BMK_display_f display_f,
        const char* filename,
        const char* functionName)
{
    if (bp.decompressOnly) {
        bp.total_bench_time_ms = 1;
        bp.notification        = 0;
    }
    BMK_timedFnState_shell tfnShell; // allocate on stack
    BMK_timedFnState_t* const tfn = BMK_initStatic_timedFnState(
            &tfnShell,
            sizeof(tfnShell),
            bp.total_bench_time_ms,
            ONE_RUN_BENCH_TIME_MS);
    if (tfn == NULL)
        exit(1);

    const size_t blockSize = bp.blockSize;
    assert(blockSize > 0);
    assert(nbBlocks > 0);
    assert(nbBlocks * blockSize <= srcSize);
    srcSize = nbBlocks * blockSize;

    const void** srcPtrs  = malloc_orDie(nbBlocks * sizeof(*srcPtrs));
    size_t* srcSizes      = malloc_orDie(nbBlocks * sizeof(*srcSizes));
    size_t* dstCapacities = malloc_orDie(nbBlocks * sizeof(*dstCapacities));

    const char* srcPtr = src;
    char* dstPtr       = dstPtrs[0];
    assert(dstPtr != NULL);
    for (size_t bn = 0; bn < nbBlocks; bn++) {
        srcPtrs[bn] = srcPtr;
        srcPtr += blockSize;
        srcSizes[bn] = blockSize;
        dstPtrs[bn]  = dstPtr;
        dstPtr += dstBlockCapacity;
        dstCapacities[bn] = dstBlockCapacity;
    }

    BMK_benchParams_t params = {
        .benchFn       = fn,
        .benchPayload  = payload,
        .initFn        = init,
        .initPayload   = payload,
        .errorFn       = NULL,
        .blockCount    = nbBlocks,
        .srcBuffers    = srcPtrs,
        .srcSizes      = srcSizes,
        .dstBuffers    = dstPtrs,
        .dstCapacities = dstCapacities,
        .blockResults  = dstSizes,
    };

    // bench
    BMK_runOutcome_t r;
    BMK_runTime_t best = { .nanoSecPerRun = ONE_RUN_BENCH_TIME_MS * 1000000 };
    int loopCount      = 0;
    while (!BMK_isCompleted_TimedFn(tfn)) {
        r = BMK_benchTimedFn(tfn, params);
        if (!BMK_isSuccessful_runOutcome(r)) {
            EXIT("Error : processing failed");
        }
        best = fasterOne(best, BMK_extract_runTime(r));
        if (bp.notification) {
            if (display_f) {
                display_f(filename, functionName, best, srcSize);
            }
            if (bp.memory) {
                size_t const programBytesSelf = getProcessMemUsage(0);
                if (programBytesSelf) {
                    printf(" ==> MaxRSS=%9zu ", programBytesSelf);
                }
            }
            printf("  \r%4u - \r", ++loopCount); // notification
            fflush(NULL);
        }
    }

    free(dstCapacities);
    free(srcSizes);
    free(srcPtrs);

    return best;
}

static size_t outSize_default(const void* src, size_t srcSize)
{
    (void)src;
    return ZL_compressBound(srcSize);
}

static const char* getFilenameFromPath(const char* filename)
{
    assert(filename != NULL);
    size_t const fsize = strlen(filename);
    for (size_t start = fsize; start > 0; start--) {
        const char c = filename[start];
        if (c == '\\' || c == '/') {
            return filename + start + 1;
        }
    }
    return filename;
}

static const char artifactFilename[] = "benchresult.bin";

static void bench_filename(
        const char* filename,
        Bench_Entry fDesc,
        void* src,
        size_t srcSize,
        BenchParams bp)
{
    if (bp.notification) {
        if (bp.decompressOnly) {
            printf("prepare %-30s for decoding \r", filename);
        } else if (fDesc.display == NULL) {
            printf("processing %-30s \r", filename);
        }
        fflush(NULL);
    }
    filename = getFilenameFromPath(filename);

    ZL_CCtx* const cctx = ZL_CCtx_create();
    assert(cctx != NULL);
    if (bp.satSet)
        (void)ZL_CCtx_setDataArena(cctx, bp.sat);
    ZL_Compressor* const cgraph = ZL_Compressor_create();
    assert(cgraph != NULL);
    ZL_DCtx* const dctx = ZL_DCtx_create();
    assert(dctx != NULL);
    if (bp.satSet)
        (void)ZL_DCtx_setStreamArena(dctx, bp.sat);
    BenchPayload payload = {
        .name   = fDesc.name,
        .graphF = fDesc.graphF,
        .cctx   = cctx,
        .cgraph = cgraph,
        .dctx   = dctx,
        // note : intParam is used as transformID in
        // zs2_decompress_transform_wrapper
        .intParam = bp.genericIntParam,
    };
    if (bp.clevel)
        (void)ZL_CCtx_setParameter(cctx, ZL_CParam_compressionLevel, bp.clevel);

    // Parameters need to be preserved across invocations (they are reset by
    // default)
    (void)ZL_CCtx_setParameter(cctx, ZL_CParam_stickyParameters, 1);

    if (fDesc.prep != NULL) {
        srcSize = fDesc.prep(src, srcSize, &payload);
    }

    size_t blockSize = bp.blockSize;
    if (blockSize > srcSize)
        EXIT("blockSize (%zu) is too large (> %zu)", blockSize, srcSize);
    if (blockSize == 0)
        blockSize = srcSize;
    bp.blockSize          = blockSize;
    size_t const nbBlocks = srcSize / blockSize;

    BMK_outSize_f const getDstCapacity =
            (fDesc.outSize == NULL) ? outSize_default : fDesc.outSize;
    size_t const dstBlockCapacity = getDstCapacity(src, blockSize);
    void* const dst               = malloc_orDie(dstBlockCapacity * nbBlocks);
    void** const dstPtrs          = malloc_orDie(nbBlocks * sizeof(*dstPtrs));
    size_t* const dstSizes        = malloc_orDie(nbBlocks * sizeof(*dstSizes));

    dstPtrs[0] = dst;
    BMK_display_f displayResult =
            (fDesc.display == NULL) ? BMK_displayResult_default : fDesc.display;
    if (bp.decompressOnly)
        displayResult = NULL;

    if (fDesc.init == NULL && fDesc.graphF != NULL)
        fDesc.init = genericGraphCreation;
    if (fDesc.func == NULL && fDesc.graphF != NULL)
        fDesc.func = genericGraphCompression;

    BMK_runTime_t const rt = bench_function(
            dstPtrs,
            dstSizes,
            nbBlocks,
            dstBlockCapacity,
            src,
            srcSize,
            fDesc.func,
            fDesc.init,
            &payload,
            bp,
            displayResult,
            filename,
            fDesc.name);

    if (bp.saveArtifact) {
        saveBuf(artifactFilename, dst, rt.sumOfReturn);
    }

    /* adjust the really benched srcSize for blockSize, for correct benchmark
     * speed measurements */
    if (blockSize)
        srcSize -= (srcSize % blockSize);

    if (!bp.decompressOnly) {
        displayResult(filename, fDesc.name, rt, srcSize);
        printf("  \n");
        fflush(NULL);
    }

    if (fDesc.graphF && !bp.noDecompress) {
        const void** const cPtrs = (void*)dstPtrs;
        BMK_runTime_t const rd   = bench_zsDecode(
                cPtrs,
                dstSizes,
                nbBlocks,
                blockSize,
                src,
                &payload,
                &bp,
                filename,
                fDesc.name);
        zsDecompressResult(filename, fDesc.name, rd, rt.sumOfReturn);
        printf("  \n");
        fflush(NULL);
    }

    free(dst);
    free(dstPtrs);
    free(dstSizes);
    ZL_DCtx_free(dctx);
    ZL_Compressor_free(cgraph);
    ZL_CCtx_free(cctx);
}

// ********************************************
// Command Line
// ********************************************

static int display_target_names(void)
{
    printf("available targets : \n");
    for (size_t fnum = 0; fnum < NB_FUNCS; fnum++) {
        printf("%s, ", scenarioList[fnum].name);
    }
    printf("\b\b  \n");
    return 0;
}

static void benchFiles(
        const char* fnTable[],
        int nbNames,
        Bench_Entry fDesc,
        BenchParams bp)
{
    for (int n = 0; n < nbNames; n++) {
        const char* const fileName = fnTable[n];
        // Load
        ZL_Buffer srcBuff    = FIO_createBuffer_fromFilename_orDie(fileName);
        void* const src      = ZL_WC_begin(ZL_B_getWC(&srcBuff));
        size_t const srcSize = ZL_B_size(&srcBuff);
        // save only the first processed file
        bp.saveArtifact &= (n == 0);

        bench_filename(fileName, fDesc, src, srcSize, bp);

        ZL_B_destroy(&srcBuff);
    }
}

/**
 * isCommand():
 * Checks if string is the same as longCommand.
 * If yes, @return 1, otherwise @return 0
 */
static int isCommand(const char* string, const char* longCommand)
{
    assert(string);
    assert(longCommand);
    size_t const comSize = strlen(longCommand);
    return !strncmp(string, longCommand, comSize);
}

/* @return: funcID (necessarily between 0 and NB_FUNCS) if present
 *          -1 on error (fname not present) */
static int funcID(const char* fname)
{
    assert(fname);
    for (size_t id = 0; id < NB_FUNCS; id++) {
        assert(scenarioList[id].name);
        if (strlen(fname) != strlen(scenarioList[id].name))
            continue;
        if (isCommand(fname, scenarioList[id].name))
            return (int)id;
    }
    return -1;
}

static int parseInt(const char* cmd)
{
    int result = 0;
    int sign   = 1;
    if (*cmd == '-') {
        sign = -1;
        ++cmd;
    }
    while (*cmd >= '0' && *cmd <= '9') {
        result *= 10;
        result += *cmd++ - '0';
    }
    return sign * result;
}

#define GET_NUM_FLAG(flagStr, SET_VAR)          \
    if (isCommand(command, flagStr)) {          \
        const char* toParse;                    \
        size_t const cmdSize = strlen(flagStr); \
        if (command[cmdSize] == '=') {          \
            toParse = command + (cmdSize + 1);  \
        } else {                                \
            argnb++;                            \
            if (argnb == argc)                  \
                return badusage(exename);       \
            toParse = argv[argnb];              \
        }                                       \
        SET_VAR;                                \
        continue;                               \
    }

#define SET_INT(i) i = parseInt(toParse);

#define GET_INT_FLAG(flagStr, intVar) GET_NUM_FLAG(flagStr, SET_INT(intVar))

#define SET_ZU(zu)                                                    \
    {                                                                 \
        int const intVar = parseInt(toParse);                         \
        if (intVar < 0)                                               \
            EXIT("parameter must be positive (%i provided)", intVar); \
        zu = (size_t)intVar;                                          \
    }

#define GET_ZU_FLAG(flagStr, zuVar) GET_NUM_FLAG(flagStr, SET_ZU(zuVar))

#define CMD_FLAG(flagStr, code)        \
    if (isCommand(command, flagStr)) { \
        code;                          \
        continue;                      \
    }

// When 2 commands have the same outcome
#define CMD_FLAG2(flagStr1, flagStr2, code)                             \
    if (isCommand(command, flagStr1) || isCommand(command, flagStr2)) { \
        code;                                                           \
        continue;                                                       \
    }

static int help(const char* exename)
{
    printf("Benchmark selected scenario on designated FILE(s) content \n\n");
    printf("Usage: %s [commands] scenario FILE(s) \n\n", exename);
    printf("Optional commands: \n");
    printf(" --list     List available Scenarios to benchmark then exit \n");
    printf("  -l=#      select compression level \n");
    printf("  -i=#      Test duration per file, in seconds \n");
    printf("  -B=#      Split input into blocks of size # bytes \n");
    printf("  -z        Compression only");
    printf(" --csv      output result in csv format \n");
    printf(" --save-result  save the 1st generated artifact into '%s' \n",
           artifactFilename);
    printf("  -h        This help \n");
    return 0;
}

static int badusage(const char* exename)
{
    printf("Error: incorrect command line \n\n");
    help(exename);
    return 1;
}

static int badfunc(const char* exename)
{
    printf("Error: incorrect target name \n\n");
    display_target_names();
    printf("\n");
    help(exename);
    return 1;
}

int main(int argc, const char* argv[])
{
    typedef enum { disp_default, disp_csv } disp_format_e;
    disp_format_e dispform = disp_default;
    assert(argc > 0);
    const char* const exename = argv[0];
    int nbSecs                = -1;
    BenchParams bp            = { .notification = 1 };

    if (argc < 2)
        return badusage(exename);
    int argnb;
    /* scan commands */
    for (argnb = 1; argnb < argc; ++argnb) {
        const char* const command = argv[argnb];

        CMD_FLAG2("-h", "--help", return help(exename));

        CMD_FLAG("--list", return display_target_names());

        CMD_FLAG2(
                "-d", "--decompress-only", bp.decompressOnly = true;
                bp.noDecompress = false)

        CMD_FLAG2(
                "-z", "--no-decompress", bp.noDecompress = true;
                bp.decompressOnly = false)

        CMD_FLAG2("-m", "--memory", bp.memory = true);

        GET_INT_FLAG("--duration_s", nbSecs);
        GET_INT_FLAG("-i", nbSecs);
        GET_INT_FLAG("-l", bp.clevel);

        GET_ZU_FLAG("--blockSize", bp.blockSize);
        GET_ZU_FLAG("-B", bp.blockSize);

        /* note: advanced hidden parameter, meaning is context-dependent */
        GET_INT_FLAG("--param", bp.genericIntParam);
        GET_INT_FLAG("-p", bp.genericIntParam);

        CMD_FLAG(
                "--stackArena", bp.sat = ZL_DataArenaType_stack; bp.satSet = 1);
        CMD_FLAG("--heapArena", bp.sat = ZL_DataArenaType_heap; bp.satSet = 1);

        CMD_FLAG("-q", bp.notification = 0);
        CMD_FLAG("--quiet", bp.notification = 0);
        CMD_FLAG(
                "--csv", dispform = disp_csv; bp.notification = 0;
                bp.noDecompress = true);

        CMD_FLAG("--save-result", bp.saveArtifact = true);

        CMD_FLAG("--", argnb++; break); // No more command after this flag

        break; // not a command => no more commmand after this point
    }

    /* must be followed by a function name */
    if (argc <= argnb)
        return badusage(exename);
    const char* const codecName = argv[argnb];
    int const fID               = funcID(codecName);
    if (fID == -1)
        return badfunc(exename);

    if (argc <= argnb + 1)
        return badusage(exename); // must have at least 1 file

    Bench_Entry fDesc = scenarioList[fID];
    if (dispform == disp_csv)
        fDesc.display = csvResult;
    if (bp.decompressOnly && fDesc.graphF == NULL)
        EXIT("wrong command : codec %s is not compatible with zstrong decoding",
             codecName);

    // Finalize benchmark parameters
    bp.total_bench_time_ms = (nbSecs >= 0) ? (unsigned)nbSecs * 1000
                                           : TOTAL_BENCH_TIME_MS_DEFAULT;
    int fileIndex          = argnb + 1;
    benchFiles(argv + fileIndex, argc - fileIndex, fDesc, bp);

    return 0;
}
