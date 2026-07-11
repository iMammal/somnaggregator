# Privacy Policy

**Effective date:** July 2026  
**Project:** SomnAggregator  
**Maintainer:** Morris Chukhman

SomnAggregator is a personal, local-first, open-source sleep and wearable data aggregation toolkit. It is designed to help users import, normalize, and analyze their own sleep and wellness data from sources such as Oura, Samsung Health, OSCAR/CPAP exports, Muse, and MindMonitor.

## Data Collection

SomnAggregator itself does not collect, transmit, sell, or share user data.

When used locally, SomnAggregator may process health, sleep, wearable, or device-export data that the user provides, including but not limited to:

- sleep duration
- sleep stages
- heart rate
- heart rate variability
- SpO2 / oxygen saturation
- respiratory rate
- CPAP usage and AHI
- Muse / MindMonitor EEG, movement, and sensor logs
- device metadata

This data remains on the user's own computer unless the user separately chooses to upload, sync, publish, or share it.

## Oura API Access

If the user authorizes SomnAggregator to access Oura data, the tool may request access to Oura API scopes such as sleep, daily summaries, heart rate, SpO2, and personal profile data.

SomnAggregator uses this access only to retrieve the user's own data for local analysis. API tokens or credentials should be stored locally by the user and should not be committed to GitHub or shared publicly.

## Local Storage

SomnAggregator may write local files such as:

- raw API JSON responses
- CSV exports
- normalized observation tables
- processed summaries
- cache files
- analysis outputs
- notebook outputs

These files may contain sensitive personal health or wellness data. Users are responsible for securing, deleting, encrypting, or backing up these files as they see fit.

Recommended private/local paths include:

- `data/raw/`
- `data/raw/api/`
- `data/interim/`
- `data/processed/`
- `outputs/`
- `.env`

These should not be committed to public repositories.

## Data Sharing

SomnAggregator does not share data with the maintainer, Dogstar Labs, Oura, Samsung, ResMed, Muse, or any third party.

Any sharing occurs only if the user chooses to do so, for example by uploading files to a cloud service, committing data to a repository, sending files to another person, or enabling third-party integrations outside of SomnAggregator.

## Analytics and Tracking

SomnAggregator does not include built-in analytics, advertising, telemetry, or tracking.

If a future hosted or desktop version adds optional telemetry, this policy should be updated before such functionality is enabled.

## Security

SomnAggregator is provided as open-source software. Users should treat sleep, wearable, and health-related exports as sensitive personal data.

Users should not commit or publish:

- API tokens
- `.env` files
- raw health data
- Oura exports
- CPAP reports
- MindMonitor CSVs
- screenshots containing personal data
- processed summaries derived from personal data

## Children's Data

SomnAggregator is not intended for use by children or for processing children's health data.

## Medical Disclaimer

SomnAggregator is not a medical device and is not intended to diagnose, treat, cure, prevent, or monitor any disease or medical condition. It is intended for personal data aggregation, wellness tracking, research prototyping, and exploratory analysis only.

Users should consult qualified medical professionals for medical advice, diagnosis, or treatment.

## Changes to This Policy

This privacy policy may be updated as the project evolves. Updates will be posted in the project repository or website.

## Contact

For questions about this policy, contact:

Morris Chukhman  
imammal@protonmail.com
