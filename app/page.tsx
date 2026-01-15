import {
  StressTestHero,
  StrategyComparison,
  ParameterSweep,
  EmbeddedSimulator,
  ScaleProof,
  GetStarted
} from '@/components/results'

export default function Home() {
  return (
    <>
      {/* Hero: 800 vehicles, zero gridlock - stress test results */}
      <StressTestHero />

      {/* 7 strategies compared side-by-side */}
      <StrategyComparison />

      {/* How tau affects cooperation - parameter sweep */}
      <ParameterSweep />

      {/* Interactive simulator - try it yourself */}
      <EmbeddedSimulator />

      {/* O(N) scaling verified - 500 to 10,000 agents */}
      <ScaleProof />

      {/* Get started - code examples */}
      <GetStarted />
    </>
  )
}
