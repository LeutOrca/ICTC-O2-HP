The private ControlPersist=600 master expired while Tavotto exports ran.
The final ordinary SSH query succeeded at 2026-11-22T13:22:47+08:00 on login02
and returned no active jobs for phy_bjxiao. The subsequent master-close command
reported an absent control socket because that task-owned master had already
expired. No other SSH session was terminated. Raw receipts are preserved.
