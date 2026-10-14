# Exact formula evaluation, not a simulation or paper reproduction.
# Target: current population cross-moment, independent homogeneous users/time.
$ErrorActionPreference = 'Stop'
$n = 10000
$epsilon = 1.0
$w = 20
$kappa = [Math]::Tanh($epsilon / (2 * $w))
$rows = foreach ($q in @(1, 5, 20)) {
    foreach ($h in @(1, 5, 20, 100)) {
        $fractionNew = [Math]::Min($q, $h) / $h
        $variance = (1 / ($kappa * $kappa) - $fractionNew * 0.36) / ($n * $h)
        $biasSquared = 0.36 * (1 - $fractionNew) * (1 - $fractionNew)
        [pscustomobject]@{
            post_change_round = $q
            statistical_window = $h
            variance = $variance
            bias_squared = $biasSquared
            mse = $variance + $biasSquared
        }
    }
}
$rows | Export-Csv -LiteralPath (Join-Path $PSScriptRoot '2026-10-14-window-error.csv') -NoTypeInformation -Encoding UTF8
$fullKappa = [Math]::Tanh($epsilon / 2)
$uniformMse = (1 / ($kappa * $kappa) - 0.36) / $n
$rotatingMse = (1 / ($fullKappa * $fullKappa) - 0.36) / ($n / $w)
[pscustomobject]@{
    uniform_every_round_mse = $uniformMse
    rotating_cohorts_mse = $rotatingMse
    ratio = $uniformMse / $rotatingMse
} | ConvertTo-Json
