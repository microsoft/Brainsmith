┌──────────────────────────────────────────────────────────────┐
│       HARDWARE MODELING ABSTRACTION LEVELS                   │
│            Fast & Simple → Slow & Detailed                   │
└──────────────────────────────────────────────────────────────┘

   Abstract (Fast) ◄────────────────────────────► Detailed (Slow)

1. ROOFLINE   ~1ms   "What's possible?"      → Math eqs, Peak limits
2. DATAFLOW   ~10ms  "How will it stream?"   → SDIM/Tiling ◄─ THIS SYSTEM, Constraints
3. C++ MODEL  ~1min  "Does algorithm work?"  → Functional, Behavioral
4. RTL SIM    ~10min "How many cycles?"      → Cycle accurate, Waveforms
5. HARDWARE   ~1hr   "Does silicon work?"    → FPGA/ASIC, Power/Timing

Each level answers different questions at different speeds!
