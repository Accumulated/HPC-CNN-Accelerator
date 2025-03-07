#include "CommonInclude.h"
#include "Timer.h"


TimerHandler::TimerHandler(bool Pause = false): Pause(Pause)
{
  /* Create the start and stop events */
  HANDLE_ERROR(cudaEventCreate(&(this -> start_timing)));
  HANDLE_ERROR(cudaEventCreate(&(this -> stop_timing)));
}


TimerHandler::~TimerHandler()
{
  /* Destroy the start and stop events */
  HANDLE_ERROR(cudaEventDestroy(this -> start_timing));
  HANDLE_ERROR(cudaEventDestroy(this -> stop_timing));
}


void TimerHandler::Timer_Start()
{
  /* Start recording */
  HANDLE_ERROR(cudaEventRecord(this -> start_timing));
}


void TimerHandler::Timer_Stop()
{
  /* Stop recording */
  HANDLE_ERROR(cudaEventRecord(this -> stop_timing));
  /* Make sure the stop event is done */
  HANDLE_ERROR(cudaEventSynchronize(this -> stop_timing));
  /* Get the elapsed time between starting and stopping the timer */
  HANDLE_ERROR(cudaEventElapsedTime(&(this -> TimeElapsed), this -> start_timing, this -> stop_timing));

  if(Pause){

    /* Incase there is a pause in between, keep track of the time in a separate
     * variable which accumulates the total time so far.
    */
    this -> TotalTime += this -> TimeElapsed;
  }

}

void TimerHandler::Timer_Report()
{

  /* When reporting the elapsed time, it depends whether there has been a pause or
   * not. If there has been a pause, then the total time is reported. Otherwise, the
   * time elapsed variable is reported.
  */
  if(Pause){

    /* Time elapsed resets after each start to the elapsed time between this current
     * start and the current stop. It's not aware of the history - use total time
     * instead.
     */
    std::cout << "Time elapsed: " << this -> TotalTime << " ms" << std::endl;

  }
  else{

    std::cout << "Time elapsed: " << this -> TimeElapsed << " ms" << std::endl;

  }

}


void TimerHandler::Timer_Reset(){

  /* Reset the total time and elapsed time */
  this -> TotalTime = 0.0;
  this -> TimeElapsed = 0.0;

}
