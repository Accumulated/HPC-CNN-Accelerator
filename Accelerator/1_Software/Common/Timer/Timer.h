#pragma once


class TimerHandler{

private:

    cudaEvent_t start_timing;
    cudaEvent_t stop_timing;
    float TimeElapsed = 0.0;
    float TotalTime = 0.0;
    bool Pause = false;

public:

    TimerHandler(bool Pause);
    ~TimerHandler();

    void Timer_Start();
    void Timer_Stop();
    void Timer_Report();
    void Timer_Reset();

};