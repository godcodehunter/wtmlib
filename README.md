# Wall-clock Time Measurement library (WTMLIB)

<sup> Port with some modification of the [WTMLIB](https://github.com/AndreyNevolin/wtmlib).</sup>\
Library supported architectures: amd64, arm.

The library allows measuring wall-clock time intervals with nanosecond precision and very low overhead (also at nanosecond scale). 

To achieve this the library uses Time Stamp Counter (TSC) and ticks conversion to nanoseconds by means of fast division-free integer arithmetic.

> [!NOTE]  
> Time Stamp Counter (TSC) - in this project this name means special microprocessor
> counter. The counter is called differently on different architectures: 
> - time-stamp counter on amd64
> - time base register on PowerPC
> - interval time counter on Itanium
> 
> end etc

> [!NOTE]
> If target's CPU manufacturer is Intel, recommended take a closer look at [Intel Processor Trace](https://halobates.de/blog/p/406)

# User guide 

The most typical scenario is the following:


TODO: 
https://habr.com/ru/articles/425237/#comment_19197483
