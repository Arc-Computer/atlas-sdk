# ExCyTIn-Bench Transparency Note

## Overview

ExCyTIn-Bench is a framework that offers code and an evaluation dataset for benchmarking models and agent implementations in cyber-security threat investigations. Designed for research purposes only, it evaluates autonomous large language model (LLM) agents in simulated environments featuring real-world attack scenarios. 

For more details, see our paper: https://arxiv.org/abs/2507.14201
## System Description

### Purpose and Functionality

**What Can ExCyTIn-Bench Do**

ExCyTIn-Bench is designed to assess the performance of LLM agents in Cyber Threat Investigation tasks, utilizing security questions derived from investigation graphs. Agents are permitted to query the simulated tenant's security event logs database through a multi-step process to address these questions. The system delivers detailed, step-by-step rewards for each action executed by agents within the environment.

### Dataset Contents

ExCyTIn-Bench features:
- **589 security questions and solutions** from logs totaling **20.4 GB**
- Each entry includes:
  - A question
  - Surrounding context
  - A step-by-step solution based on the investigation graph
  - A label marking relevant data points
- **Log data collection**: August 2024
- **Q&A generation**: September 2024
- **External data sources**: None linked

## Intended Uses

ExCyTIn-Bench provides a platform for the research community, such as researchers in industry and academia, to benchmark threat investigation agents. It enables users to:

- Analyze and compare different models and agentic techniques
- Assess performance on various aspects of threat investigation
- Support reproducibility and encourage additional studies in this field

**Target Users**: Domain experts who are able to evaluate output quality prior to taking action.

## Out-of-Scope Uses

### What ExCyTIn-Bench Is NOT

- **Not a universal tool** for evaluating cyber-security threat investigations across all environments
- **Not for commercial or real-world applications** - released for research purposes only
- **Not for highly regulated domains** where inaccurate outputs could suggest actions that lead to injury or negatively impact legal, financial, or life opportunities
- **Not for high-risk decision making** contexts (e.g., law enforcement, legal, finance, or healthcare)

### Important Limitations

- The dataset lacks user-specific attack scenarios
- Provides general performance benchmark, not environment-specific
- Inherent bias towards threat investigation and specific attack scenarios
- Developers should evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use

## Data and Privacy

### Data Creation & Processing

ExCyTIn-Bench was constructed using:
- **Synthetic attack data** generated from the Microsoft demo tenant "Alpine Ski House"
- **Collection period**: August 2024
- **Environment management**: Alpine Ski House team at Microsoft

### Privacy Protection

Due to potential presence of sensitive information (IP addresses, account names), a robust **PII anonymization pipeline** was employed:
- Manual verification processes
- Large language model review
- Additional methodology details available in the referenced paper

### People & Identifiers

- **Individual data**: All data points relating to individual characteristics are entirely synthetic
- **Children's data**: ExCyTIn-Bench does not contain any data about children
- **PII removal**: Potentially identifiable information removed through manual methods and LLM-based review
- **Design principle**: Avoids information that could directly or indirectly identify individuals

### Sensitive Content Safeguards

**Data did NOT include**:
- Racial or ethnic background
- Sexual orientation
- Religious affiliation
- Disability status
- Political views
- Financial or health records
- Biometric or genetic details
- Criminal history
- Sexual material, violence, hate speech, or self-harm references

## Performance and Limitations

### Evaluation Results

Comprehensive experiments confirm the difficulty of cyber-security threat investigation tasks:
- **Average reward across all evaluated models**: 24.9%
- **Best performance achieved**: 36.8%
- **Substantial headroom** remains for future research

### Known Limitations

1. **Research and experimental purposes only** - further testing needed for commercial applications
2. **English language only** - performance in other languages may vary and requires expert assessment
3. **AI-generated outputs** may include factual errors, fabrication, or speculation
4. **Human oversight required** - decisions should not be based solely on system outputs
5. **Model inheritance** - inherits biases, errors, or omissions from base models
6. **Default evaluation model**: GPT-4o (alternative models may yield different results)
7. **Security vulnerabilities** - no systematic effort to protect against indirect prompt injection attacks

### Bias Considerations

- **No systematic evaluation** for sociocultural/economic/demographic/linguistic bias
- **Training data bias** may be amplified by AI-generated interpretations
- **Model selection importance** - developers should carefully choose appropriate base LLM/Multimodal LLM

## Evaluation Methodology

### Validation Process

A team of security researchers and experts manually reviewed random question and answer samples to confirm:
- Validity of questions and answers
- Solvability with available security event logs
- Further validation details provided in the referenced paper

### Evaluation Methods

- **Fine-grained rewards** (normalized quantifiable reward signals for reinforcement learning)
- **Per-step assessment** rather than overall sequence scoring
- **Comparative analysis** among various LLMs and agentic methodologies
- **Default evaluator**: GPT-4o (see Table 2 in paper for model comparisons)

**Note**: Benchmarks may change when using different LLM models for evaluation

## Best Practices

### Getting Started

To begin using ExCyTIn-Bench, see instructions in the repository: [microsoft/SecRL: Benchmarking LLM agents on Cyber Threat Investigation](https://github.com/microsoft/SecRL)

### Recommended Practices

1. **Data splitting**: Use "minimizing path overlap" strategy as described in the referenced paper
2. **LLM selection**: Use LLMs/MLLMs with robust responsible AI mitigations (e.g., Azure OpenAI services)
3. **Safety mitigations**: Employ services that continually update safety and RAI mitigations

### Responsible AI Resources

- [Blog post on responsible AI features in AOAI (Ignite 2023)](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/announcing-new-ai-safety-amp-responsible-ai-features-in-azure/ba-p/3983686)
- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview)
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI's Usage policies](https://openai.com/policies/usage-policies)
- [Azure OpenAI's Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct)

### Legal and Regulatory Considerations

Users must evaluate potential specific legal and regulatory obligations when using AI services and solutions. AI services may not be appropriate for use in every industry or scenario and are not designed for ways prohibited in applicable terms of service and relevant codes of conduct.

## Licensing and Trademarks

### License
MIT License

### Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to Microsoft's Trademark & Brand Guidelines. Use in modified versions must not cause confusion or imply Microsoft sponsorship. Third-party trademarks or logos are subject to those third-party's policies.

## Contact Information

### Research Team
This research was conducted by members of **Microsoft Models Research**.

### Contact Details
- **Primary Contact**: amudgerikar@microsoft.com
- **Secondary Contact**: ashleyto@microsoft.com

### Feedback and Collaboration
We welcome feedback and collaboration from our audience. If you have:
- Suggestions or questions
- Observations of unexpected/offensive behavior
- Reports of undesired behavior/content

Please contact us using the email addresses above.

### Issue Resolution
If the team receives reports of undesired behavior/content or identifies issues independently, we will update the repository with appropriate mitigations.

## Document Information

- **Version**: 1.0
- **Last Updated**: August 20, 2025
- **Document Type**: Research Transparency Note
- **Purpose**: Research and experimental use only

---

*This transparency note is a living document and will be updated as the system evolves and new information becomes available. For the most current information, please refer to the referenced paper: [2507.14201] ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation*