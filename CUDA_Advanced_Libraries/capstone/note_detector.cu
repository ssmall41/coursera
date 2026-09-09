// nvcc -O2 note_detector.cu -o note_detector -lcufft -lsndfile

#include <cuda_runtime.h>
#include <cufft.h>
#include <sndfile.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Error-checking helpers
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

#define CUFFT_CHECK(call)                                                     \
    do {                                                                     \
        cufftResult err = (call);                                            \
        if (err != CUFFT_SUCCESS) {                                          \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__    \
                      << " -> code " << err << std::endl;                    \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)


// Convert a frequency in Hz to the closest musical note.
std::string freq_to_note(double frequency)
{
    if (frequency <= 0.0)
        throw std::invalid_argument("Frequency must be positive");

    static const char* note_names[12] = {"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"};

    const double A4 = 440.0;

    // Number of semitones away from A4 (can be fractional)
    double semitones_from_a4 = 12.0 * std::log2(frequency / A4);
    long nearest_semitone = std::lround(semitones_from_a4);

    // MIDI note number: A4 = 69
    long midi_number = 69 + nearest_semitone;

    // Octave and note index (MIDI: C-1 = 0, so octave = midi//12 - 1)
    long octave = static_cast<long>(std::floor(static_cast<double>(midi_number) / 12.0)) - 1;
    long note_index = ((midi_number % 12) + 12) % 12;

    std::ostringstream oss;
    oss << note_names[note_index] << octave;
    return oss.str();
}

// Given a real-valued time series and the sample rate, find the dominant frequency
double find_dominant_frequency(const std::vector<float>& signal, int samplerate, cufftHandle plan)
{
    const int N = static_cast<int>(signal.size());
    if (N <= 0)
        throw std::invalid_argument("Signal must be non-empty");

    // Allocate device memory
    float* d_input = nullptr;
    cufftComplex* d_output = nullptr;
    const int N_complex = N / 2 + 1; // R2C output size

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(cufftComplex) * N_complex));
    CUDA_CHECK(cudaMemcpy(d_input, signal.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    // Create and execute the cuFFT plan
    //cufftHandle plan;
    //CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftExecR2C(plan, d_input, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the (relevant) spectrum back to the host
    std::vector<cufftComplex> h_output(N_complex);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, sizeof(cufftComplex) * N_complex, cudaMemcpyDeviceToHost));

    // Find the dominant frequency in the first N//2 bins
    const int half = N / 2;
    int max_idx = 0;
    double max_mag = -1.0;
    for (int i = 0; i < half; ++i)
    {
        double re = h_output[i].x;
        double im = h_output[i].y;
        double mag = std::sqrt(re * re + im * im);
        if (mag > max_mag)
        {
            max_mag = mag;
            max_idx = i;
        }
    }

    double max_freq = static_cast<double>(max_idx) * static_cast<double>(samplerate) / static_cast<double>(N);

    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return max_freq;
}


// Read an audio file with libsndfile and return the first (left) channel as a vector<float>, and the sample rate.
std::vector<float> readMonoChannel(const std::string& path, int& samplerate)
{
    SF_INFO sfinfo;
    std::memset(&sfinfo, 0, sizeof(sfinfo));

    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        std::cerr << "Error opening file " << path << ": " << sf_strerror(nullptr) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    samplerate = sfinfo.samplerate;
    const sf_count_t frames = sfinfo.frames;
    const int channels = sfinfo.channels;

    std::vector<float> interleaved(static_cast<size_t>(frames) * channels);
    sf_count_t readCount = sf_readf_float(file, interleaved.data(), frames);
    sf_close(file);

    if (readCount != frames) {
        std::cerr << "Warning: expected " << frames << " frames, read " << readCount
                   << " from " << path << std::endl;
    }

    // Extract channel 0
    std::vector<float> mono(static_cast<size_t>(readCount));
    for (sf_count_t i = 0; i < readCount; ++i) {
        mono[static_cast<size_t>(i)] = interleaved[static_cast<size_t>(i) * channels];
    }

    return mono;
}

// For each note, slice samples [5000:6000) of channel 0, run find_dominant_frequency, convert to a note name, and print.
void scanInstrument(const std::string& label, const std::vector<std::string>& notes, 
		    const std::vector<std::vector<float>>& data, const std::vector<int>& samplerates)
{
    std::cout << "Scanning the " << label << " data........................" << std::endl;

    // Define limits on the signal. Use something in the middle to avoid edge problems.
    const size_t start = 5000;
    const size_t end = 6000;
    const int N = static_cast<int>(signal.size());

    // Create the cuFFT plan here
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));

    for (size_t i = 0; i < notes.size(); ++i)
    {
        const std::vector<float>& full = data[i];
        const int samplerate = samplerates[i];

        if (full.size() < end) {
            std::cerr << "  Warning: signal for note " << notes[i]
                      << " is shorter than the requested slice [5000:6000)." << std::endl;
            continue;
        }

        std::vector<float> segment(full.begin() + start, full.begin() + end);

        double estimated_freq = find_dominant_frequency(segment, samplerate, plan);
        std::string estimated_note = freq_to_note(estimated_freq);

        std::cout << "For actual note " << notes[i] << ", estimating " << estimated_note << "." << std::endl;
    }
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------
int main() {
    const std::vector<std::string> notes = {
        "C5", "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5"
    };

    // --- Load Flute (no vibrato) data ---
    std::vector<std::vector<float>> data_flute_nonvib;
    std::vector<int> samplerate_flute_nonvib;
    for (const auto& n : notes) {
        int sr = 0;
        std::string path = "data/Flute.nonvib.ff." + n + ".stereo.aif";
        data_flute_nonvib.push_back(readMonoChannel(path, sr));
        samplerate_flute_nonvib.push_back(sr);
    }

    // --- Load Oboe data ---
    std::vector<std::vector<float>> data_oboe;
    std::vector<int> samplerate_oboe;
    for (const auto& n : notes) {
        int sr = 0;
        std::string path = "data/Oboe.ff." + n + ".stereo.aif";
        data_oboe.push_back(readMonoChannel(path, sr));
        samplerate_oboe.push_back(sr);
    }

    // --- Load Trumpet (no vibrato) data ---
    std::vector<std::vector<float>> data_trumpet_nonvib;
    std::vector<int> samplerate_trumpet_nonvib;
    for (const auto& n : notes) {
        int sr = 0;
        std::string path = "data/Trumpet.novib.ff." + n + ".stereo.aif";
        data_trumpet_nonvib.push_back(readMonoChannel(path, sr));
        samplerate_trumpet_nonvib.push_back(sr);
    }

    // --- Iterate over the data and find the estimated notes ---
    scanInstrument("Flute (no vibrato)", notes, data_flute_nonvib, samplerate_flute_nonvib);
    scanInstrument("Oboe", notes, data_oboe, samplerate_oboe);
    scanInstrument("Trumpet (no vibrato)", notes, data_trumpet_nonvib, samplerate_trumpet_nonvib);

    return 0;
}
