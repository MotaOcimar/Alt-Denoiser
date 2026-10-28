#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "DeepFilterNetProcessor.h"
#include "Resampler.hpp"
#include <vector>
#include <memory>
#include <atomic>
#include <thread>

// Single-producer / single-consumer ring buffer, lock-free (juce::AbstractFifo)
class SpscFifo {
public:
    // not thread-safe: call only while the worker thread is stopped
    void setSize(int size) {
        buffer.assign(size, 0.0f);
        fifo.setTotalSize(size);
    }

    void clear() {
        fifo.reset();
    }

    int push(const float* data, int numSamples) {
        const auto scope = fifo.write(numSamples);
        if (scope.blockSize1 > 0) std::copy(data, data + scope.blockSize1, buffer.data() + scope.startIndex1);
        if (scope.blockSize2 > 0) std::copy(data + scope.blockSize1, data + scope.blockSize1 + scope.blockSize2, buffer.data() + scope.startIndex2);
        return scope.blockSize1 + scope.blockSize2;
    }

    int pushSilence(int numSamples) {
        const auto scope = fifo.write(numSamples);
        if (scope.blockSize1 > 0) std::fill_n(buffer.data() + scope.startIndex1, scope.blockSize1, 0.0f);
        if (scope.blockSize2 > 0) std::fill_n(buffer.data() + scope.startIndex2, scope.blockSize2, 0.0f);
        return scope.blockSize1 + scope.blockSize2;
    }

    int pop(float* dest, int numSamples) {
        const auto scope = fifo.read(numSamples);
        if (scope.blockSize1 > 0) std::copy_n(buffer.data() + scope.startIndex1, scope.blockSize1, dest);
        if (scope.blockSize2 > 0) std::copy_n(buffer.data() + scope.startIndex2, scope.blockSize2, dest + scope.blockSize1);
        return scope.blockSize1 + scope.blockSize2;
    }

    int getAvailable() const { return fifo.getNumReady(); }

private:
    std::vector<float> buffer;
    juce::AbstractFifo fifo { 1 };
};

class AltDenoiserProcessor : public juce::AudioProcessor {
public:
    AltDenoiserProcessor();
    ~AltDenoiserProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void reset() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Alt Denoiser"; }
    
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override
    {
        auto state = apvts.copyState();
        std::unique_ptr<juce::XmlElement> xml(state.createXml());
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

        if (xmlState.get() != nullptr)
        {
            if (xmlState->hasTagName(apvts.state.getType()))
            {
                apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
            }
        }
    }

    std::atomic<float> inputRmsLevel { 0.0f };
    std::atomic<float> outputRmsLevel { 0.0f };
    juce::AudioProcessorValueTreeState apvts;

private:
    std::unique_ptr<DeepFilterNetProcessor> dfProcessor;

    // inference runs on its own thread, so a slow frame uses the latency headroom
    // instead of stalling the host's audio thread
    void startWorker();
    void stopWorker();
    void workerLoop();

    SpscFifo inputFifo;
    SpscFifo outputFifo;
    std::vector<float> tempInputFrame;
    std::vector<float> tempOutputFrame;
    bool modelLoaded = false;
    float lastAttenLim = -1.0f;

    std::thread worker;
    std::atomic<bool> workerShouldStop { false };
    juce::WaitableEvent inputReady;
    juce::WaitableEvent framesDone;
    std::atomic<float> attenLim { 100.0f };
    juce::int64 samplesPushed = 0;               // audio thread only
    std::atomic<juce::int64> framesProcessed { 0 };

    std::unique_ptr<Resampler<1, 1>> resamplerHandler;
    std::vector<float> resampleInBuffer;
    std::vector<float> resampleOutBuffer;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AltDenoiserProcessor)
};