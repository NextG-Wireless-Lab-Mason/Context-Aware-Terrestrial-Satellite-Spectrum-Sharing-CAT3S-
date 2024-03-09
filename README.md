# Context-Aware-Terrestrial-Satellite-Spectrum-Sharing-(CAT3S)- Case Study of 12 GHz Band

#Project Overview:

This project proposes a holistic, multi-disciplinary, context-aware spectrum-sharing approach to address the coexistence of broadband wireless systems in satellite bands. We have developed a novel context-aware (e.g., weather conditions- sunny and rainy) spectrum-sharing framework, "CAT3S," to enable spectrum coexistence of 5G broadband in the 12 GHz satellite bands. This framework considers a realistic deployment scenario that includes the fixed satellite services (FSS) receiver located at 1770 Forecast Drive, Blacksburg, Virginia, the exact positions of all 33 Macro Base Stations (MBS) from OpenCellID within a 5000m circular coverage area of the FSS receiver, and geolocation data of 8,644 buildings from OpenStreetMap. Additionally, the context acquisition unit gathers weather data, interference thresholds for different weather conditions, exclusion zone radii, and different transmitting power levels of MBSs from the context information providers. Consequently, the simulator begins analyzing the interference for each MBS, employing industry-standardized codebook-based beamforming at 5G MBSs, directional signal reception at FSS receivers, channel scheduling at 5G MBSs, path loss between interfering MBS and FSS, and noise power. Moreover, the aggregate interference-to-noise ratio is calculated for each MBS, and then the CAT3S algorithm is utilized. This context-aware MBS control algorithm then determines whether a particular base station is on or off and, if on, determines its corresponding optimal transmitting power and operational beam for each base station sector. This approach aims to maximize the network throughput of the 5G network while keeping the overall interference at the incumbent receivers below an acceptable threshold.


#How to Run the code:

1. First, gather all required data by accessing the contents within the "Codebook" folder and the "data" folder or utilize the split_csv.py script provided in the "Context-aware-Spectrum-Coexistence-Analyzer" project [1].
3. Run the rest_server.py which will send the necessary information to the simulator.py. For setting up the Dynamic Spectrum Access (DSA) framework's environment, ensure to incorporate all the information from the "DSA" folder located in the "Context-aware-Spectrum-Coexistence-Analyzer" [1].
4. Run the simulator.py which will analyze the I/N for all the MBSs and dynamically optimize the multi-antenna MBSs’ parameters using a polynomial time complexity MBS control algorithm.
5. Run the app.py

#Reference:

[1] Project - "Context-aware-Spectrum-Coexistence-Analyzer"- URL: https://github.com/NextG-Wireless-Lab-Mason/Context-aware-Spectrum-Coexistence-Analyzer.

#Acknowledgement:

This project is generously supported by the National Science Foundation under Award # 2128584. Find more information about the project here: https://www.nextgwirelesslab.org/current-projects/spectrum-sharing-in-satellite-bands

#Relevant Publications

[1] T. Niloy, Z. Hassan, R. Smith, V. Anapana, and V. K. Shah, Context-Aware Spectrum Coexistence of Terrestrial Beyond 5G Networks in Satellite Bands, IEEE Conference of Dynamic Spectrum Access Networks (DySPAN), 2024

[2] T. Niloy, S. Kumar, A. Hore, Z. Hassan, E. Burger, C. Dietrich, J. Reed, and V. K. Shah, ASCENT: A Context-Aware Spectrum Coexistence Design and Implementation Toolset for Policymakers in Satellite Bands, IEEE Conference of Dynamic Spectrum Access Networks (DySPAN), 2024. Link: https://arxiv.org/abs/2402.05273

[3] T. R. Niloy, Z. Hassan, N. Stephenson, and V. K. Shah, Interference Analysis of Coexisting 5G Networks and NGSO FSS Receivers in the 12 GHz Band, IEEE Wireless Comms. Letters (WCL), 2023 Link: https://ieeexplore.ieee.org/document/10139318
