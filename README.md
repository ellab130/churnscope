## ChurnScope — Business-Focused Churn Decision Support Tool

ChurnScope is an end-to-end churn analytics demo that turns ML predictions into an actionable retention policy under real operational constraints. Instead of stopping at model metrics, the project answers the business question: **“Given limited outreach capacity, who should we contact to reduce churn, and will it pay off?”**

The tool ranks customers by predicted churn risk, simulates contacting the top-N highest-risk customers, and estimates **incremental profit** using unit economics (contact cost, value at risk from churn, and campaign save rate). It also includes side-by-side strategy comparisons (model targeting vs random targeting vs do-nothing), scenario presets, and an interactive **capacity → profit curve** to help identify a practical operating point.

### Key features
- **Capacity-based targeting policy:** contact the top-N customers by churn probability
- **Business metrics:** estimated profit, total cost, ROI, break-even save rate
- **Strategy comparison:** Model vs Random vs Do Nothing (same capacity)
- **Sensitivity analysis:** adjustable contact cost, churn loss (LTV at risk), and save rate
- **Explainable outputs:** executive summary + business interpretation
- **Deployable demo:** static site (GitHub Pages) that loads scored data from a CSV (no backend required)

### Profit model (incremental ROI)
`profit = expected_TP × (save_rate × churn_loss) − contacts × contact_cost`

This frames churn modeling as a decision-support problem: using model ranking to allocate limited retention resources efficiently.
