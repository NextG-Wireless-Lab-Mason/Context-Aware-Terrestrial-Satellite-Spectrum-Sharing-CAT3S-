# Context-Aware-Terrestrial-Satellite-Spectrum-Sharing-(CAT3S)- Case Study of 12 GHz Band

#Project Overview:

This project proposes a holistic, multi-disciplinary, context-aware spectrum-sharing approach to address the spectrum coexistence of broadband wireless systems in satellite bands. In this work, we developed a novel context-aware (i.e. weather-sunny & rainy) spectrum-sharing framework "CAT3S" for enabling spectrum coexistence of 5G broadband in 12 GHz satellite bands by considering a realistic deployment of the FSS receiver (1770 Forecast Drive, Blacksburg, Virginia), the exact position of all 33 MBS (Macro Base Station) from opencellID within a 5000m circular area, and Geolocation data of 8644 Buildings from OpenStreetMap. The DSA framework will transmit exclusion zone and weather details, base station information, and interference thresholds for both sunny and rainy weather to the simulator. Therefore, the simulator will start analyzing the interference for each MBS based on the industry-standardized codebookbased beamforming at 5G MBSs, directional signal reception at FSS receivers, channel scheduling at 5G MBSs, Path Loss between Interfering MBS and FSS, and Noise power. Moreover, the aggregate interference-to-noise ratio is calculated for each MBS and then the CAT3S algorithm is utilized which comprises two key components –(i) a context-acquisition unit to capture critical contexts for spectrum sharing, and (ii) a context-aware MBS control unit that determines whether a particular BS is on/off, and if on, then determines its corresponding optimal transmitting power and operational beam at each BS sector, such that the network throughput of the 5G network is maximized while maintaining the overall interference below an acceptable threshold at the incumbent receiver(s). 


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
