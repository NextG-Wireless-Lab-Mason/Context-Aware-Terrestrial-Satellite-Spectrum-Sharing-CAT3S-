"""
Simulator REST API for CCI-SWIFT ASCENT

Created: Jan 29, 2023
Author/s: Rishabh Rastogi (rrishabh@vt.edu), Ta Seen Reaz Niloy (tniloy@gmu.edu)
Advised by Dr. Eric Burger (ewburger@vt.edu) and Dr. Vijay K Shah (vshah22@gmu.edu)
For SWIFT-ASCENT
"""

# !/usr/bin/env python
import cmath
import json
import math
import os
import pickle
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List
import matplotlib
import matplotlib.pyplot as plt
import base64
import io
import copy
import numpy as np
import pandas as pd
from Geometry3D import *
from flask import Flask, request, jsonify
from matplotlib.lines import Line2D
from scipy import optimize
from shapely import geometry
from tqdm import tqdm
from weather import get_weather
import warnings
import time
import time
from matplotlib.offsetbox import AnchoredText

warnings.filterwarnings("ignore")
matplotlib.use('Agg')



def get_weather_json(latitude, longitude):
    weather_info = get_weather(f"{latitude},{longitude}")
    if weather_info:
        print('Temperature:', weather_info['main']['temp'])
        print('Humidity:', weather_info['main']['humidity'])
        print('Weather:', weather_info['weather'][0]['description'])
        return weather_info
    else:
        print('Could not retrieve weather information for the given location')


def get_exclusion_zone_x_parameter(keys, x_pos_real):
    for i, key in enumerate(keys):
        if key > x_pos_real:
            if i == 0:
                return -1
            else:
                return i - 0.5
        elif key == x_pos_real:
            return i
    return len(keys) + 1


def path_loss_UMi(BS_X, BS_Y, BS_Z, FSS_X, FSS_Y, FSS_Z, ctx):
    ##UMi
    ##LOS,SF=4:

    buildings = ctx.buildings
    saved_los = ctx.saved_los
    ##(10m<=d_2D)<=D_BP:
    # RR - these variables needed as input
    h = 5
    d_2D = math.sqrt(
        ((FSS_X - BS_X) ** 2) + ((FSS_Y - BS_Y) ** 2) + ((FSS_Z - BS_Z) ** 2)
    )
    hBs = 10
    fc = 12
    d_3D = math.sqrt(hBs ** 2 + d_2D ** 2)
    PL1umi = 32.4 + 21 * math.log10(d_3D) + 20 * math.log10(fc)

    ##(D_BP<=d_2D)<=5000m:
    hBs = 10
    hUT = 4.5
    hE = 1
    hBs1 = hBs - hE
    hUT1 = hUT - hE
    c = 3e8
    D_BP = (4 * hBs1 * hUT1 * 12e9) / c
    PL2umi = (
            32.4
            + 40 * math.log10(d_3D)
            + 20 * math.log10(fc)
            - 9.5 * math.log10((D_BP) ** 2 + (hBs - hUT) ** 2)
    )

    PLUMiLOS = 0
    PL1umiNLOS = 0
    # weather = get_weather_json(latitude=ctx.lat_FSS, longitude=ctx.lon_FSS)
    # rain = weather["rain"]
    # x = weather["rain"]["1h"] #rain_rate
    rain = ctx.rain
    x = ctx.rain_rate

    if rain:
        # x is the rain rate mm/h
        P = -5.520 * 10 ** -12 * x ** 3 + 3.26 * 10 ** -9 * x ** 2 - 1.21 * x * 10 ** -7 - 6 * 10 ** -6  # av (considering vertical polarization)
        Q = 8 * 10 ** -10 * x ** 3 - 4.552 * 10 ** -7 * x ** 2 - 3.03 * x * 10 ** -5 + 0.001  # bv (considering vertical polarization)
        R = -5.71 * 10 ** -9 * x ** 3 + 6 * 10 ** -7 * x ** 2 + 8.707 * x * 10 ** -3 - 0.018  # cv (considering vertical polarization)
        S = - 1.073 * 10 ** -7 * x ** 3 + 1.068 * 10 ** -4 * x ** 2 - 0.0598 * x + 0.0442  # dv (considering vertical polarization)

        # Attenuation Factor due to rain
        A = (P * (fc ** 3) + Q * (fc ** 2) + R * fc + S) / 1000  # (dB/m)

        if 10 <= d_2D and d_2D <= D_BP:
            PLUMiLOS = PL1umi + A
        elif D_BP <= d_2D and d_2D <= 5000:
            PLUMiLOS = PL2umi + A
        else:
            PLUMiLOS = 1 + A

            ##NLOS,SF=7.82:
        PL1umiNLOS = (
                             35.3 * math.log10(d_3D) + 22.4 + 21.3 * math.log10(fc) - 0.3 * (hUT - 1.5)
                     ) + A
    else:

        if 10 <= d_2D and d_2D <= D_BP:
            PLUMiLOS = PL1umi
        elif D_BP <= d_2D and d_2D <= 5000:
            PLUMiLOS = PL2umi
        else:
            PLUMiLOS = 1

        ##NLOS,SF=7.82:
        PL1umiNLOS = (
                35.3 * math.log10(d_3D) + 22.4 + 21.3 * math.log10(fc) - 0.3 * (hUT - 1.5)
        )

    PLUMiNLOS = max(PLUMiLOS, PL1umiNLOS)

    line_of_sight = True

    for i in tqdm(range(len(buildings))):
        for polygon in buildings[i].wall_polygons:
            coordinates = ((BS_X, BS_Y, BS_Z), (FSS_X, FSS_Y, FSS_Z), *[(p.x, p.y, p.z) for p in polygon.points])
            # polygon_hash = hash(polygon)
            # if (bs_to_fss_segment_hash, polygon_hash) in saved_los:
            if coordinates in saved_los:
                if saved_los.get(coordinates):
                    path_loss_UMi = PLUMiNLOS
                    line_of_sight = False
                else:
                    path_loss_UMi = PLUMiLOS
            else:
                bs_to_fss_segment = Segment(Point(BS_X, BS_Y, BS_Z), Point(FSS_X, FSS_Y, FSS_Z))
                # bs_to_fss_segment_hash = hash(bs_to_fss_segment)
                # polygon_hash = hash(polygon)
                if intersection(bs_to_fss_segment, polygon) is not None:
                    path_loss_UMi = PLUMiNLOS
                    line_of_sight = False
                    saved_los[coordinates] = True
                else:
                    path_loss_UMi = PLUMiLOS
                    saved_los[coordinates] = False


    ##realistic pathloss:
    # bs_to_fss_segment = Segment(Point(BS_X, BS_Y, BS_Z), Point(FSS_X, FSS_Y, FSS_Z))
    # for i in tqdm(range(len(buildings))):
    #     #     for i in range(len(buildings)):
    #     for polygon in buildings[i].wall_polygons:
    #         if (hash(bs_to_fss_segment), hash(polygon)) in saved_los:
    #             if saved_los.get((hash(bs_to_fss_segment), hash(polygon))):
    #                 path_loss_UMi = PLUMiNLOS
    #                 line_of_sight = False
    #             else:
    #                 path_loss_UMi = PLUMiLOS
    #         #                     line_of_sight = True
    #
    #         # intersection(line, polygon): [Intersection], None
    #         elif intersection(bs_to_fss_segment, polygon) is not None:
    #             path_loss_UMi = PLUMiNLOS
    #             line_of_sight = False
    #             saved_los[(hash(bs_to_fss_segment), hash(polygon))] = True
    #         else:
    #             path_loss_UMi = PLUMiLOS
    #             #                 line_of_sight = True
    #             saved_los[(hash(bs_to_fss_segment), hash(polygon))] = False

    # realisticpathlossmodified
    # bs_to_fss_segment = Segment(Point(BS_X, BS_Y, BS_Z), Point(FSS_X, FSS_Y, FSS_Z))
    # bs_to_fss_segment_hash = hash(bs_to_fss_segment)
    #
    # all_polygons = [(i, polygon) for i, building in enumerate(buildings) for polygon in building.wall_polygons]
    #
    # for i, polygon in tqdm(all_polygons):
    #     polygon_hash = hash(polygon)
    #     if (bs_to_fss_segment_hash, polygon_hash) in saved_los:
    #         if saved_los.get((bs_to_fss_segment_hash, polygon_hash)):
    #             path_loss_UMi = PLUMiNLOS
    #             line_of_sight = False
    #         else:
    #             path_loss_UMi = PLUMiLOS
    #     elif intersection(bs_to_fss_segment, polygon) is not None:
    #         path_loss_UMi = PLUMiNLOS
    #         line_of_sight = False
    #         saved_los[(bs_to_fss_segment_hash, polygon_hash)] = True
    #     else:
    #         path_loss_UMi = PLUMiLOS
    #         saved_los[(bs_to_fss_segment_hash, polygon_hash)] = False

    return path_loss_UMi, d_2D, line_of_sight, saved_los

    # #Loss Probability:
    #     if d_2D>18:
    #         PrLosUmi=(18/d_2D)+math.exp((-(d_2D/36))*(1-(18/d_2D)))
    #     else:
    #         PrLosUmi=1 ##d_2d<=18
    #     if PrLosUmi > random.random():
    #         path_loss_UMi = PLUMiLOS
    #     else:
    #         path_loss_UMi = PLUMiNLOS
    ##path_loss
    #     path_loss_UMi= PL2umi*PrLosUmi+PL1umiNLOS*(1-PrLosUmi)

    # return path_loss_UMi, d_2D, line_of_sight


# In[ ]:


def path_loss_UMa(BS_X, BS_Y, BS_Z, FSS_X, FSS_Y, FSS_Z, ctx):
    ##LOS,SF=4:
    ##(10m<=d_2D)<=D_BP:
    # RR - these variables needed as input
    buildings = ctx.buildings
    saved_los = ctx.saved_los
    fc = 12
    h = 5
    d_2D = math.sqrt(((FSS_X - BS_X) ** 2) + (FSS_Y - BS_Y) ** 2) + (
            (FSS_Z - BS_Z) ** 2
    )
    hBs = 25
    d_3D = math.sqrt(hBs ** 2 + d_2D ** 2)
    PL3uma = 28.0 + 22 * math.log10(d_3D) + 20 * math.log10(fc)

    ##(D_BP<=d_2D) <=5000m:
    hUT = 4.5
    hE = 1
    hBs1 = hBs - hE
    hUT1 = hUT - hE
    c = 3 * 10 ** 8
    D_BP = (4 * hBs1 * hUT1 * fc) / c
    PL4uma = (
            28.0
            + 40 * math.log10(d_3D)
            + 20 * math.log10(fc)
            - 9 * math.log10((D_BP) ** 2 + (hBs - hUT) ** 2)
    )

    if 10 <= d_2D and d_2D <= D_BP:
        PLUMALOS = PL3uma
    elif D_BP <= d_2D and d_2D <= 5000:
        PLUMALOS = PL4uma
    else:
        PLUMALOS = 1

    ##NLOS,SF=6:
    PL1NLOSuma = (
            13.54 + 39.08 * math.log10(d_3D) + 20 * math.log10(fc) - 0.6 * (hUT - 1.5)
    )
    PLUMANLOS = max(PLUMALOS, PL1NLOSuma)

    line_of_sight = True

    # realisticpathlossmodified
    bs_to_fss_segment = Segment(Point(BS_X, BS_Y, BS_Z), Point(FSS_X, FSS_Y, FSS_Z))
    bs_to_fss_segment_hash = hash(bs_to_fss_segment)

    all_polygons = [(i, polygon) for i, building in enumerate(buildings) for polygon in building.wall_polygons]

    for i, polygon in tqdm(all_polygons):
        polygon_hash = hash(polygon)
        if (bs_to_fss_segment_hash, polygon_hash) in saved_los:
            if saved_los.get((bs_to_fss_segment_hash, polygon_hash)):
                path_loss_UMa = PLUMANLOS
                line_of_sight = False
            else:
                path_loss_UMa = PLUMALOS
        elif intersection(bs_to_fss_segment, polygon) is not None:
            path_loss_UMa = PLUMANLOS
            line_of_sight = False
            saved_los[(bs_to_fss_segment_hash, polygon_hash)] = True
        else:
            path_loss_UMa = PLUMALOS
            saved_los[(bs_to_fss_segment_hash, polygon_hash)] = False

    return path_loss_UMa, d_2D, line_of_sight

    ##NLOS,SF=7.8 (optional)
    ##PL_Optional=32.4+20*math.log(fc)+30*math.log(d_3D)

    ##Loss Probability:
    #     if d_2D >18:
    #         if hUT<=13:
    #             ChUT=0
    #         elif 13<hUT and hUT<=23:
    #             ChUT=((hUT-13)/10)**1.5
    #         PrLOSUma=((18/d_2D)+math.exp((-(d_2D/63))*(1-(18/d_2D))))*(1+ChUT*(5/4)*((d_2D/100)**3)*math.exp(-(d_2D/150)))
    #     else:##18m<d_2D:
    #         PrLOSUma=1

    #     if PrLOSUma > random.random():
    #         path_loss_UMa = PLUMALOS

    #     else:
    #         path_loss_UMa = PLUMANLOS

    ##Path Loss
    #     path_loss_UMa= PL4uma*PrLOSUma+PL1NLOSuma*(1-PrLOSUma)
    # return path_loss_UMa, d_2D, line_of_sight


# In[ ]:


def path_loss_RMa(BS_X, BS_Y, BS_Z, FSS_X, FSS_Y, FSS_Z, ctx):
    buildings = ctx.buildings
    saved_los = ctx.saved_los
    ##LOS,SF=4(PL1),SF=6(PL2)
    ##10m<=d_2D<=d_BP:
    # RR - these variables needed as input
    fc = 12
    h = 5
    d_2D = math.sqrt(
        ((FSS_X - BS_X) ** 2) + ((FSS_Y - BS_Y) ** 2) + ((FSS_Z - BS_Z) ** 2)
    )
    hBs = 35
    d_3D = math.sqrt(hBs ** 2 + d_2D ** 2)
    PL1rma = (
            20 * math.log10((40 * math.pi * d_3D * fc) / 3)
            + min(0.03 * h ** 1.72, 10) * math.log10(d_3D)
            - min(0.044 * h ** 1.72, 14.77)
            + 0.002 * math.log10(h) * d_3D
    )

    ##d_BP<=d_2D<=10km:
    hUT = 4.5
    c = 3 * 10 ** 8
    d_BP = (2 * math.pi * hBs * hUT * fc) / c
    PL2rma = PL1rma * (d_BP) + 40 * math.log10(d_3D / d_BP)

    if 10 <= d_2D and d_2D <= d_BP:
        PLRMALOS = PL1rma
    elif d_BP <= d_2D and d_2D <= 10000:
        PLRMALOS = PL2rma
    else:
        PLRMALOS = 1

    ##NLOS,SF=8:
    W = 20
    h = 5
    PL1NLOSrma = (
            161.04
            - 7.11 * math.log10(W)
            + 7.5 * math.log10(h)
            - (24.37 - 3.7 * (h / hBs) ** 2) * math.log10(hBs)
            + (43.42 - 3.1 * math.log10(hBs)) * (math.log10(d_3D) - 3)
            + 20 * math.log10(fc)
            - (3.2 * (math.log10(11.75 * hUT)) ** 2 - 4.97)
    )
    PLRMANLOS = max(PLRMALOS, PL1NLOSrma)

    line_of_sight = True
    # realisticpathlossmodified
    bs_to_fss_segment = (Point(BS_X, BS_Y, BS_Z), Point(FSS_X, FSS_Y, FSS_Z))
    bs_to_fss_segment_hash = hash(bs_to_fss_segment)

    all_polygons = [(i, polygon) for i, building in enumerate(buildings) for polygon in building.wall_polygons]

    for i, polygon in tqdm(all_polygons):
        polygon_hash = hash(polygon)
        if (bs_to_fss_segment_hash, polygon_hash) in saved_los:
            if saved_los.get((bs_to_fss_segment_hash, polygon_hash)):
                path_loss_RMa = PLRMANLOS
                line_of_sight = False
            else:
                path_loss_RMa = PLRMALOS
        elif intersection(bs_to_fss_segment, polygon) is not None:
            path_loss_RMa = PLRMANLOS
            line_of_sight = False
            saved_los[(bs_to_fss_segment_hash, polygon_hash)] = True
        else:
            path_loss_RMa = PLRMALOS
            saved_los[(bs_to_fss_segment_hash, polygon_hash)] = False

    return path_loss_RMa, d_2D, line_of_sight



# In[ ]:

global Pmin, Pmax, beam_scores
beam_scores = []
Pmin = int(4)
Pmax = int(10)
def Interface_UMi_1(
        BS_X,
        BS_Y,
        BS_Z,
        FSS_X,
        FSS_Y,
        FSS_Z,
        FSS_phi,
        pathloss_UMi,
        theta_tilt,
        phi_scan,
        num_subarrays,
        interface_UMi_W_subarray,
        codebook_weights,
        Noise_W,
        output=False,
):
    LBodyLoss = 4
    #         LSpectralOverlap=10*math.log(10)
    # theta_tilt, phi_scan = max_gain_5g_parameters(theta, phi)
    x, y, z = BS_X - FSS_X, BS_Y - FSS_Y, (10 - 4.5)

    theta_bs_es = math.degrees(math.atan(y / x)) % 360
    phi_bs_es = math.degrees(math.sqrt(x ** 2 + y ** 2) / z) % 360
    fss_phi_difference = abs(FSS_phi - phi_bs_es)
    if output:
        print("theta_bs_es:", theta_bs_es, "phi_bs_es:", phi_bs_es)

    # G_5G_R = gain_5g(theta_bs_es, phi_bs_es, theta_tilt, phi_scan)
    G_5G_R, G_5G_R_subarray = gain_5g(theta_bs_es, phi_bs_es, theta_tilt, phi_scan)
    # print("G_5G_R_subarray", G_5G_R_subarray)
    G_Rx_5G = gain_fss_wbes_b(fss_phi_difference)

    # TXPower = -6.77
    #         LBuildingLoss=
    # Pmin = int(-6.7)
    # Pmax = int(1)
    k = 1.38064852 * 10 ** (-23)
    T = 200
    B = 240e6
    Noise = 10 * math.log10(k * T * B)
    Noise_W = 10 ** (Noise / 10)

    interface1 = {}
    for p in range(Pmin, Pmax+1):
        # interface1 = p + G_5G_R - pathloss_UMi - LBodyLoss + G_Rx_5G
        interface1[p] = p + G_5G_R - pathloss_UMi - LBodyLoss + G_Rx_5G


    interface_subarray = {}
    SNR = {}
    start_time_for2 = time.time()
    # G_5G_R_subarray = 0.01599558028614669
    for p in range(Pmin, Pmax+1):
        interface_subarray[p] = {}
        SNR[p] = {}
        SNR[p]= 10 * np.log(p) + G_5G_R - pathloss_UMi - 10 * np.log(Noise_W) - 10 * np.log(num_subarrays)
        for subarray in G_5G_R_subarray:
            interface_subarray[p][subarray] = {}
            for theta in range(0, 360):
                for phi in range(0, 360):
                    interface_subarray[p][subarray][(theta,phi)] = p + G_5G_R_subarray[subarray][(theta,phi)] - pathloss_UMi - LBodyLoss + G_Rx_5G - 10 * np.log(num_subarrays)
    end_time_for2 = time.time()
    print(f"Time for nested loops: {end_time_for2 - start_time_for2} seconds")

    Iagg = {}
    # start_time_for3 = time.time()
    for p in range(Pmin, Pmax+1):
        Iagg[p] = {}
        for subarray in interface_subarray[p]:
            Iagg[p][subarray] = interface_subarray[p][subarray]
    # end_time_for3 = time.time()
    # print(f"Time for Iagg loop: {end_time_for3 - start_time_for3} seconds": 0.0 seconds)

    # print("asol Iagg dekhte chai", Iagg)

    return interface1, pathloss_UMi, interface_subarray, SNR, Iagg




def compute_score(SNR, Iagg, codebook_weights, beam_scores):
    global num_subarrays
    constant_weight_parameter= int (0.9)
    max_score = None
    best_codebook_weight = None
    best_power = None
    best_subarray = None
    capacity_full =[]

    for p in range(Pmin, Pmax + 1):
        capacity = np.log2(1 + 10 ** (SNR[p] / 10))
        capacity_full.append(capacity)
    total_capacity = np.sum(capacity_full)
    # print(total_capacity, "Capacity")

    # start_time_scoring = time.time()
    for p in Iagg:
        for subarray in Iagg[p]:
            # print("second loop ", subarray)
            for theta, phi in codebook_weights[subarray - 1]:
                # print("Third loop ", theta, phi)
                User_per_beam = beam_scores[p][(theta, phi)]
                # print("MOne hoy ", User_per_beam)
                numerator = (1 - constant_weight_parameter) * total_capacity + (
                            constant_weight_parameter * User_per_beam)
                # print("numerator eikhane print hobe", numerator)
                denominator = Iagg[p][subarray]  # [(theta,phi)]
                # print("denominator eikhane print hobe", denominator)
                score = numerator / denominator
                if max_score is None or score > max_score:
                    max_score = score
                    best_codebook_weight = (theta, phi)
                    best_power = p
                    best_subarray = subarray
    # end_time_scoring = time.time()
    # # Print the time taken for the main scoring loop
    # print("Time for scoring loop:", end_time_scoring - start_time_scoring, "seconds")

    # print("Maximum Score:", max_score)
    # print("Best Codebook Weight:", best_codebook_weight)
    # # print('best_power', best_power)
    # print('best_subarray', best_subarray)

    return max_score, best_codebook_weight, best_power, best_subarray

def compute_score_1(SNR, Iagg, codebook_weights, beam_scores):
    global num_subarrays
    constant_weight_parameter= int (0.9)
    max_score = None
    best_codebook_weight = None
    best_power = None
    best_subarray = None
    capacity_full =[]

    start_time_capacity1 = time.time()
    for p in range(Pmin, Pmax + 1):
        capacity = np.log2(1 + 10 ** (SNR[p] / 10))
        capacity_full.append(capacity)
    total_capacity = np.sum(capacity_full)
    end_time_capacity1 = time.time()
    # Print the time taken for capacity calculation
    print("Time for capacity1 calculation:", end_time_capacity1 - start_time_capacity1, "seconds")
    # print(total_capacity, "Capacity")

    start_time_scoring1 = time.time()
    for p in Iagg:
        for subarray in Iagg[p]:
            # print(codebook_weights, "Debugging codebook_weights:")
            # print(subarray, codebook_weights[subarray - 1], "Debugging subarray index and value:")
            for theta, phi in codebook_weights[subarray - 1]:
                User_per_beam = beam_scores[p][(theta, phi)]
                numerator = (1 - constant_weight_parameter) * total_capacity + (
                            constant_weight_parameter * User_per_beam)
                # print("numerator eikhane print hobe", numerator)
                denominator = Iagg[p][subarray]  # [(theta,phi)]
                # print("denominator eikhane print hobe", denominator)
                score = numerator / denominator
                if max_score is None or score > max_score:
                    max_score = score
                    best_codebook_weight = (theta, phi)
                    best_power = p
                    best_subarray = subarray
    end_time_scoring1 = time.time()
    # Print the time taken for the main scoring loop
    print("Time for scoring1 loop:", end_time_scoring1 - start_time_scoring1, "seconds")

    return max_score, best_codebook_weight, best_power, best_subarray


def choose_base_stations(SNR, interface_UMi_W_subarray,codebook_weights_all,base_station_count,beam_scores):
    global num_subarrays, Ith
    Boperational = []
    max_score = -np.inf
    best_bs = None
    best_power = None
    best_subarray = None
    best_codebook_weight = None

    for bs in range(base_station_count):
    # if (bs, best_subarray, best_power) not in Balready_chosen:
        score, codebook_weight, power, subarray = compute_score_1(
            SNR,
            interface_UMi_W_subarray,
            codebook_weights_all,
            beam_scores[bs]
        )
        if score > max_score:
            max_score = score
            best_bs = bs
            best_power = power
            best_subarray = subarray
            best_codebook_weight = codebook_weight

        if best_bs is None:
            break

    print("the values of interface_UMi_W_subarray", interface_UMi_W_subarray)
    Iagg = interface_UMi_W_subarray
    for p in Iagg:
        for subarray in Iagg[p]:
            if Iagg[p][subarray] > Ith:
                break

        Boperational.append((best_bs, best_subarray, best_power))


    print("operational BS", Boperational)

    return Boperational


def Interface_UMa_1(
        BS_X,
        BS_Y,
        BS_Z,
        FSS_X,
        FSS_Y,
        FSS_Z,
        FSS_phi,
        pathloss_UMa,
        theta_tilt,
        phi_scan,
        output=False,
):
    LBodyLoss = 4
    #         LSpectralOverlap=10*math.log(10)
    x, y, z = BS_X - FSS_X, BS_Y - FSS_Y, BS_Z - FSS_Z

    theta_bs_es = math.degrees(math.atan(y / x)) % 360
    phi_bs_es = math.degrees(math.sqrt(x ** 2 + y ** 2) / z) % 360
    fss_phi_difference = abs(FSS_phi - phi_bs_es)
    # if output: print("theta_bs_es:", theta_bs_es, "phi_bs_es:", phi_bs_es)

    G_5G_R = gain_5g(theta_bs_es, phi_bs_es, theta_tilt, phi_scan)
    G_Rx_5G = gain_fss_wbes_b(fss_phi_difference)


    TXPower = -6.77
    #         LBuildingLoss=1
    interface2 = TXPower + G_5G_R - pathloss_UMa - LBodyLoss + G_Rx_5G
    return interface2, pathloss_UMa


def Interface_RMa_1(
        BS_X,
        BS_Y,
        BS_Z,
        FSS_X,
        FSS_Y,
        FSS_Z,
        FSS_phi,
        pathloss_RMa,
        theta_tilt,
        phi_scan,
        output=False,
):
    LBodyLoss = 4
    #         LSpectralOverlap=10*math.log(10)
    x, y, z = BS_X - FSS_X, BS_Y - FSS_Y, BS_Z - FSS_Z

    theta_bs_es = math.degrees(math.atan(y / x)) % 360
    phi_bs_es = math.degrees(math.sqrt(x ** 2 + y ** 2) / z) % 360
    # if output: print("theta_bs_es:", theta_bs_es, "phi_bs_es:", phi_bs_es)
    fss_phi_difference = abs(FSS_phi - phi_bs_es)
    if output:
        print("fss_phi_difference:", fss_phi_difference)

    G_5G_R = gain_5g(theta_bs_es, phi_bs_es, theta_tilt, phi_scan)
    G_Rx_5G = gain_fss_wbes_b(fss_phi_difference)

    TXPower = -6.77
    #         LBuildingLoss=1
    interface3 = TXPower + G_5G_R - pathloss_RMa - LBodyLoss + G_Rx_5G
    return interface3, pathloss_RMa


def simulate(output=True, ctx=None):
    global codebooks, codebook_weights_all, Ith, Array_AF, beam_scores, Iagg, SNR

    FSS_X = np.array([])
    FSS_Y = np.array([])
    FSS_Z = np.array([])
    FSS_CHANNELS = []
    x = ctx.x
    y = ctx.y
    z = ctx.z
    FSS_phi = ctx.FSS_phi
    Noise_W = ctx.Noise_W
    data_within_zone = ctx.data_within_zone
    base_station_count = ctx.base_station_count
    radius = ctx.radius
    R = ctx.R
    if ctx.rain:
        Ith = 56.234  #-12
    else:
        Ith = 141.254   #-8.5


    for i in range(1):
        FSS_X = np.append(FSS_X, x)
        FSS_Y = np.append(FSS_Y, y)
        FSS_Z = np.append(FSS_Z, 4.5)
        # 0 means not in use, 1 means in use
        channel_status = [
            random.randint(0, 1) for i in range(FSS_Channels.channel_count)
        ]
        channels_used = np.array(
            [i for i in range(FSS_Channels.channel_count) if channel_status[i] == 1]
        )
        FSS_CHANNELS.append(channels_used)
        if output:
            print(
                "FSS Co-ordinates="
                + str(x)
                + ","
                + str(y)
                + ","
                + str(z)
                + ", channel: "
                + str(channels_used)
            )
    if output:
        print(FSS_X, FSS_Y, FSS_Z, FSS_CHANNELS)

    # Create base stations
    BS_X = np.array([])
    BS_Y = np.array([])
    BS_Z = np.array([])

    num_sectors = 3
    num_UEs_per_sector = 10

    # Randomly select base stations
    # base_station_indexes = random.sample(range(len(data_within_zone)), base_station_count)

    # for i in base_station_indexes:
    for i in range(base_station_count):
        lat_BS, lon_BS = (
            data_within_zone.iloc[i]["latitude"],
            data_within_zone.iloc[i]["longitude"],
        )
        # bs1 = BS(radius, max_height=35, carr_freq=12e3, interference_type=None)
        x_BS = R * math.cos(math.radians(lat_BS)) * math.cos(math.radians(lon_BS))
        y_BS = R * math.cos(math.radians(lat_BS)) * math.sin(math.radians(lon_BS))
        #         z_BS = R * math.sin(math.radians(lat_BS))
        z_BS = 10

        x_FSS = ctx.x_FSS
        y_FSS = ctx.y_FSS
        bs_ue_min_radius = ctx.bs_ue_min_radius
        bs_ue_max_radius = ctx.bs_ue_max_radius

        BS_X = np.append(BS_X, x_BS - x_FSS)
        BS_Y = np.append(BS_Y, y_BS - y_FSS)
        BS_Z = np.append(BS_Z, 10)
        if output:
            print("Bs Co-ordinates=" + str(x_BS) + "," + str(y_BS) + "," + str(z_BS))

    if output:
        print(BS_X, BS_Y, BS_Z)

    # Create user equipment
    UE_X = np.array([])
    UE_Y = np.array([])
    UE_Z = np.array([])
    UE_CHANNEL = np.array([])
    for p in range(len(BS_X)):
        for i in range(num_sectors):
            # number of split regions
            # i is the sector number
            for j in range(num_UEs_per_sector):
                # number of UEs per region
                # j is the number of the UE in one sector
                bs_x, bs_y, bs_z = BS_X[p], BS_Y[p], BS_Z[p]
                theta_bs_ue = random.uniform(120 * i, 120 * (i + 1))
                # 0-120, 120-240, 240-360
                radius_bs_ue = random.uniform(bs_ue_min_radius, bs_ue_max_radius)

                x1 = bs_x + radius_bs_ue * math.cos(math.radians(theta_bs_ue))
                y1 = bs_y + radius_bs_ue * math.sin(math.radians(theta_bs_ue))

                UE_X = np.append(UE_X, x1)
                UE_Y = np.append(UE_Y, y1)
                UE_Z = np.append(UE_Z, 1.5)

                maximum_UEs_per_channel = 4

                if j > maximum_UEs_per_channel * BS_Channels.channel_count:
                    raise Exception(f"BS cannot support {j} UEs")

                count = {i: 0 for i in range(1, BS_Channels.channel_count + 1)}

                channel = random.randint(1, BS_Channels.channel_count)

                while count[channel] >= maximum_UEs_per_channel:
                    channel = random.randint(1, BS_Channels.channel_count)

                count[channel] += 1

                UE_CHANNEL = np.append(UE_CHANNEL, channel)

                if output:
                    print(
                        "UE Co-ordinates="
                        + str(x1)
                        + ","
                        + str(y1)
                        + ", channel: "
                        + str(channel)
                    )
            if output:
                print(UE_X, UE_Y, UE_Z)

    pathloss_UMa = np.empty([0])
    pathloss_UMi = np.empty([0])
    pathloss_RMa = np.empty([0])
    distance_UMa = np.empty([0])
    distance_UMi = np.empty([0])
    distance_RMa = np.empty([0])
    line_of_sight = np.empty([0])
    for i in range(len(BS_X)):
        for j in range(len(FSS_X)):
            pathlossumi, distance, los_single, ctx.saved_los = path_loss_UMi(
                BS_X[i], BS_Y[i], 10, FSS_X[j], FSS_Y[j], 4.5, ctx
            )
            distance = data_within_zone.iloc[i]["dist_from_FSS"]
            pathlossuma = []
            pathlossrma = []
            # pathlossuma, distance, los_single = path_loss_UMa(
            #     BS_X[i], BS_Y[i], 25, FSS_X[j], FSS_Y[j], FSS_Z[j], ctx
            # )
            # pathlossrma, distance, los_single = path_loss_RMa(
            #     BS_X[i], BS_Y[i], 35, FSS_X[j], FSS_Y[j], FSS_Z[j], ctx
            # )
            if output:
                print(
                    "pathloss umi:",
                    pathlossumi,
                    "uma:",
                    pathlossuma,
                    "rma:",
                    pathlossrma,
                    "for distance",
                    distance,
                )

            pathloss_UMa = np.append(pathloss_UMa, pathlossuma)
            distance_UMa = np.append(distance_UMa, distance)
            pathloss_UMi = np.append(pathloss_UMi, pathlossumi)
            distance_UMi = np.append(distance_UMi, distance)
            pathloss_RMa = np.append(pathloss_RMa, pathlossrma)
            distance_RMa = np.append(distance_RMa, distance)

            line_of_sight = np.append(line_of_sight, los_single)

    # print( "distance_UMi", distance_UMi)

    if output:
        print(pathloss_UMi, distance_UMi)
    if output:
        print(pathloss_UMa, distance_UMa)
    if output:
        print(pathloss_RMa, distance_RMa)


    # interface_UMi_W = []
    # interface_UMi_W_subarray = []
    interface_UMa_W = np.empty([0])
    interface_RMa_W = np.empty([0])
    score_UMi = {}
    global num_subarrays
    num_subarrays = 12
    best_codebook_weights = []

    if Array_AF is None or len(Array_AF) == 0:
        # print('if e dhukse')
        Array_AF = generate_beam_patterns(base_station_count, num_sectors, num_UEs_per_sector)
        with open('array_af.pkl',"wb") as f:
            pickle.dump(Array_AF, f)

    interface_UMi_W = {power: {} for power in range(Pmin, Pmax + 1)}
    # interface_UMi_W = [{power: {} for power in range(Pmin, Pmax + 1)} for i in range(len(BS_X))]
    # interface_UMi_W = [{power: {} for power in range(Pmin, Pmax)} for i in range(num_of_elements)]
    interface_UMi_W_subarray = {power: {} for power in range(Pmin, Pmax + 1)}
    # interface_UMi_W_subarray = {power: {} for power in range(Pmin, Pmax + 1)}
    interface_UMi_BS_subarray = {power: {} for power in range(Pmin, Pmax + 1)}
    interface_UMi_BS = {power: {} for power in range(Pmin, Pmax + 1)}
    umi_w_array = {}
    # umi_w_subarray_list ={}
    for power in range(Pmin, Pmax+1):
        for i in range(len(BS_X)):
            # interface_UMi_W.append({power: {} for power in range(Pmin, Pmax+1)})
            # # interface_UMi_W = [{power: {} for power in range(Pmin, Pmax)} for i in range(num_of_elements)]
            # interface_UMi_W_subarray.append({power: {} for power in range(Pmin, Pmax+1)})
            # interface_UMi_BS_subarray = {power: {} for power in range(Pmin, Pmax+1)}
            # interface_UMi_BS = {power: {} for power in
            # range(Pmin, Pmax+1)}
            # interface_UMi_BS = {power: value for power, value in enumerate(interfaceumi)}

            # interface_UMi_W_subarray[power][i] = {}
            interface_UMi_BS_subarray[power][i] = {}

            interface_UMa_BS = np.empty([0])
            interface_RMa_BS = np.empty([0])


            for j in range(len(FSS_X)):
                if output:
                    print(f"BS {i}, FSS {j}, pathloss {pathloss_UMi[i * len(FSS_X) + j]}")
                for k in random.sample(range(len(UE_X)), 30):
                    # print(f"UE{k}")
                    # channel check
                    # if UE is using channel 1, the start is 12.2GHz and the end is 12.3GHz
                    bs_channel_start, bs_channel_end = BS_Channels.getChannelRange(
                        UE_CHANNEL[k]
                    )
                    bs_channel_range = range(
                        bs_channel_start, int(bs_channel_end + (5e6)), int(5e6)
                    )

                    interference_found = False

                    for fss_channel in FSS_CHANNELS[j]:
                        # if fss_channel >= 6:
                        fss_channel_start, fss_channel_end = FSS_Channels.getChannelRange(
                            fss_channel
                        )
                        fss_channel_range = range(
                            bs_channel_start, int(bs_channel_end + (5e6)), int(5e6)
                        )

                        bs_set = set(bs_channel_range)
                        if len(bs_set.intersection(fss_channel_range)):
                            interference_found = True

                    if not interference_found:
                        continue

                    for interference_type in ["UMi", "UMa", "RMa"]:
                        if interference_type == "UMi":
                            BS_Z = np.array([10 for i in range(len(BS_X))])
                        elif interference_type == "UMa":
                            BS_Z = np.array([25 for i in range(len(BS_X))])
                        elif interference_type == "RMa":
                            BS_Z = np.array([35 for i in range(len(BS_X))])
                        # bs_ue_x, bs_ue_y, bs_ue_z = BS_X-UE_X, BS_Y-UE_Y, BS_Z-UE_Z
                        bs_ue_x, bs_ue_y, bs_ue_z = (
                            UE_X[k] - BS_X[i],
                            UE_Y[k] - BS_Y[i],
                            UE_Z[k] - BS_Z[i],
                        )

                        theta_bs_ue = np.arctan(bs_ue_y / bs_ue_x)
                        phi_bs_ue = np.sqrt(bs_ue_x ** 2 + bs_ue_y ** 2) / bs_ue_z

                        theta_bs_ue = np.degrees(theta_bs_ue) % 360
                        phi_bs_ue = np.degrees(phi_bs_ue) % 360
                        if output:
                            print("theta_bs_ue:", theta_bs_ue, "phi_bs_ue:", phi_bs_ue)

                        theta_tilt, phi_scan = max_gain_5g_parameters(
                            theta_bs_ue, phi_bs_ue, ctx
                        )
                        theta_tilt = 10

                        if interference_type == "UMi":
                            interfaceumi, pathloss_UMi_x, interface_subarray, SNR, Iagg = Interface_UMi_1(
                                BS_X[i],
                                BS_Y[i],
                                BS_Z[i],
                                FSS_X[j],
                                FSS_Y[j],
                                FSS_Z[j],
                                FSS_phi['UMi'],
                                pathloss_UMi[i * len(FSS_X) + j],
                                theta_tilt,
                                phi_scan,
                                num_subarrays,
                                interface_UMi_W_subarray,
                                codebook_weights_all,
                                Noise_W,
                            )
                            if output:
                                print("UE:", k, "/ interference umi:", interfaceumi, "/ pathloss:", pathlossumi)
                            # if output:
                            #     print(
                            #         "interference umi:",
                            #         interfaceumi,
                            #         "pathloss:",
                            #         pathlossumi,
                            #     )
                            # print(Iagg)

                            for power in interface_UMi_BS:
                                interface_UMi_BS[power]= interfaceumi[power]
                                interface_UMi_BS_subarray[power] ={}
                                for subarray in Iagg[power]:
                                    interface_UMi_BS_subarray[power][subarray] = Iagg[power][subarray]

                            # print( interface_UMi_BS[power], "Saaalalallala")
            for power in interface_UMi_BS.keys():
                interface_UMi_W [power]= 10**(interface_UMi_BS[power]/10)
              #  interface_UMi_W[power] = np.sum(10 ** (interface_UMi_BS[power] / 10))
                interface_UMi_W_subarray[power]= {}
                for subarray in range(1, num_subarrays+1):
                    # print(interface_UMi_BS_subarray[power][subarray])
                    interface_UMi_W_subarray[power][subarray] = 0
                    for phitheta in interface_UMi_BS_subarray[power][subarray-1].keys():
                        interface_UMi_W_subarray[power][subarray]+= 10 ** (interface_UMi_BS_subarray[power][subarray-1][phitheta] / 10)
                    #interface_UMi_W_subarray[power][subarray] = 10 ** (interface_UMi_BS_subarray[power][subarray] / 10)

            # for power in interface_UMi_W:
            #     interface_UMi_W[power] = np.sum(10 ** (interface_UMi_BS[power] / 10))
            #     # interface_UMi_W_subarray[power] = {}
            #     for subarray in interface_UMi_W_subarray[power]:
            #         interface_UMi_W_subarray[power][subarray] = np.sum(10 ** (interface_UMi_BS[power][subarray] / 10))
            #
            #

            if beam_scores is None or Array_AF is None or len(beam_scores) == 0:
                # print('if e dhukse')
                beam_scores = generate_beam_scores(base_station_count, UE_X, UE_Y, UE_Z, num_sectors, num_UEs_per_sector)
                with open('beam_scores.pkl', "wb") as f:
                    pickle.dump(beam_scores, f)



            # umi_w_array[i] = interface_UMi_W
            umi_w_array[i] = copy.deepcopy(interface_UMi_W)
            # print('interface_UMi_W eikhane', umi_w_array[i])

            # print('codebook_weights_all eikhane dekha jabe', codebook_weights_all)
            # print('beam_scores eikhane dekha jabe', beam_scores[i])
            # umi_w_subarray_list[i]= copy.deepcopy(interface_UMi_W_subarray)



            # find the best codebook weight for a single base station
            max_score, best_codebook_weight, best_power, best_subarray = compute_score(
                SNR,
                # interface_UMi_W_subarray[0][power][i],
                interface_UMi_W_subarray,
                codebook_weights_all,
                beam_scores[i]
            )

            if output:
                print(f"Best codebook weight for BS {i}: {best_codebook_weight}")

            best_codebook_weights.append(best_codebook_weight)

            # interface_UMi_W = np.append(
            #     interface_UMi_W, np.sum(10 ** (interface_UMi_BS / 10))

            interface_UMa_W = np.append(
                interface_UMa_W, np.sum(10 ** (interface_UMa_BS / 10))
            )
            interface_RMa_W = np.append(
                interface_RMa_W, np.sum(10 ** (interface_RMa_BS / 10))
            )
        break
    # print('interface_UMi_W array value', umi_w_array)

    if output:
        print(interface_UMi_W, pathloss_UMi)
    if output:
        print(interface_UMi_W_subarray, pathloss_UMi)

    if output:
        print(interface_UMa_W, pathloss_UMa)
    if output:
        print(interface_RMa_W, pathloss_RMa)


    I_N_umi_np = np.zeros((33, 7))
    for i in umi_w_array:
        # I_N_UMi[i] ={}
        for power in umi_w_array[i]:
            I_N_umi_np[i][power - 4] = umi_w_array[i][power] / Noise_W

    # I_N_UMi = I_N_umi_np

    if output:
        print("I/N UMi in watt:", I_N_umi_np)

    # print('interference to Noise ratio in watt', I_N_umi_np)

    I_N_UMi = 10 * np.log10(I_N_umi_np)
    print('interference to Noise ratio in dB', I_N_UMi)

    # I_N_UMi = {}
    # for power in interface_UMi_W_rearranged:
    #     I_N_UMi[power] = {}
    #     for i in interface_UMi_W_rearranged[power]:
    #         # Perform the division with a tolerance check to handle very small values
    #         if abs(Noise_W) < tolerance:
    #             # Handle the case when Noise_W is very close to 0 or 0
    #             I_N_UMi[power][i] = float('inf')  # or any other suitable value
    #         else:
    #             I_N_UMi[power][i] = interface_UMi_W_rearranged[power][i] / Noise_W
    # if output:
    #     print("I/N UMi:", I_N_UMi)


    # print('interference to Noise ratio', I_N_UMi)


    operational_BS = choose_base_stations(
        SNR,
        interface_UMi_W_subarray,
        codebook_weights_all,
        len(BS_X),
        ctx.beam_scores
    )

    if output:
        print(operational_BS, BS_X)


    I_N_UMa = interface_UMa_W / Noise_W
    if output:
        print("I/N UMa:", I_N_UMa)
    I_N_RMa = interface_RMa_W / Noise_W
    if output:
        print("I/N RMa:", I_N_RMa)

    return [
        distance_RMa,
        I_N_RMa,
        distance_UMa,
        I_N_UMa,
        distance_UMi,
        I_N_UMi,
        line_of_sight,
        ctx.saved_los,
        interface_UMi_W_subarray,
        score_UMi,
        operational_BS,
    ]


def gain_antenna_element_horizontal(phi) -> float:
    phi_3db = 80  # degrees
    front_to_back_ratio = 30  # dB
    return -min(12 * (phi / phi_3db) ** 2, front_to_back_ratio)


# antenna vertical pattern
def gain_antenna_element_vertical(theta) -> float:
    theta_3db = 65  # degrees
    side_lobe_level_limit = 30  # dB
    return -min(12 * ((theta - 90) / theta_3db) ** 2, side_lobe_level_limit)


# antenna element gain of elevation and azimuth plane
def gain_antenna_element(theta, phi) -> float:
    front_to_back_ratio = 30  # dB
    antenna_gain_max = 8
    return antenna_gain_max - min(
        -(gain_antenna_element_horizontal(phi) + gain_antenna_element_vertical(theta)),
        front_to_back_ratio,
    )


def superposition(n, m, theta, phi, hspace, vspace) -> complex:
    return cmath.exp(
        complex(
            0,
            2
            * math.pi
            * (
                    (n - 1) * vspace * math.cos(math.radians(theta))
                    + (m - 1)
                    * hspace
                    * math.sin(math.radians(theta))
                    * math.sin(math.radians(phi))
            ),
        )
    )


def weighting(n, m, theta_tilt, phi_scan, hspace, vspace, rows, cols) -> complex:
    return cmath.exp(
        complex(
            0,
            -2
            * math.pi
            * (
                    (n - 1) * vspace * math.sin(math.radians(theta_tilt))
                    + (m - 1)
                    * hspace
                    * math.cos(math.radians(theta_tilt))
                    * math.sin(math.radians(phi_scan))
            ),
        )
    ) / cmath.sqrt(rows * cols)


# returns theta_tilt and phi_scan which yield maximum antenna gain given theta and phi
def max_gain_5g_parameters(theta, phi, ctx, coarse=True, rounding_precision=0) -> tuple:
    saved_tp = ctx.saved_tp
    if coarse:
        theta = round(theta, rounding_precision)
        phi = round(phi, rounding_precision)
    if (theta, phi) in saved_tp:
        # print(f'match found for ({theta}, {phi}), using that')
        return saved_tp.get((theta, phi))

    # return max_parameters
    # scipy's optimization can only find the minimum, so we pass a function which returns the negative of the weighting function
    result = optimize.brute(
        lambda x: - beam_pattern_5g(
            theta, phi, x[0],  x[0], x[1])[0],
         # x[0] = theta_tilt, x[1] = phi_scan
        # theta_tilt is between -90 and 90 degrees, phi_scan is between -180 and 180 degrees
        ranges=[(-90, 90), (-180, 180)],
    )
    saved_tp[(theta, phi)] = tuple(x for x in result)

    return saved_tp[(theta, phi)]


# a_A, the directional pattern from beam forming with an array of elements



max_beam_angles = {}
max_watts_5g = 0.01599558028614669
CODEBOOK_ROUNDING_DECIMAL_PLACES = 0
def beam_pattern_5g(theta, phi, theta_tilt, phi_scan, rounding_precision=CODEBOOK_ROUNDING_DECIMAL_PLACES):
    global Array_AF, codebook_weights_all, beam_scores
    # print("Array_AF", Array_AF)
    watts = 0
    watt_subarray= {}
    max_watts_5g = 0.01599558028614669 #in watt which is 12.04 dB a max gain for max beam angle
    for subarray in codebook_weights_all:
        if (int(theta), int(phi)) in codebook_weights_all[subarray]:
        # if int(phi) in max_beam_angles[subarray][int(phi)]:
            watts += max_watts_5g
        else:
            # print( subarray, Array_AF[subarray].keys())
            watts += Array_AF[subarray][(int(round(theta)) % 360, int(round(phi)) % 360)]
        watt_subarray[subarray] = {}
        for _theta in range(0, 360):
            for _phi in range(0, 360):
                if (int(theta), int(phi)) in codebook_weights_all:
                    watt_subarray[subarray][(_theta,_phi)] = max_watts_5g
                else:
                    watt_subarray[subarray][(_theta,_phi)] = Array_AF[subarray][(int(round(theta)) % 360,int(round(phi)) % 360)]
    # print("subarray gain in watt")#, watt_subarray)
   # print(watt_subarray[12].keys())
    return watts, watt_subarray




def load_datasets(codebook_file):
    if isinstance(codebook_file, str):
        phases_input = np.loadtxt(codebook_file, delimiter=",", skiprows=1)
    else:
        phases_input = codebook_file
    # theta, phi: list of phases
    phase_values = {(int(round(col[0], CODEBOOK_ROUNDING_DECIMAL_PLACES)),
                     int(round(col[1], CODEBOOK_ROUNDING_DECIMAL_PLACES))): col[2:] for col in phases_input}
    # phase_values = {(round(col[0], CODEBOOK_ROUNDING_DECIMAL_PLACES), round(col[1], CODEBOOK_ROUNDING_DECIMAL_PLACES)): col[2:] for col in phases_input}

    return phase_values

def UE_random_co_ordinates():
    UE_x = random.randint(-4000, 4000) + 1000 * random.random()
    UE_y = random.randint(-4000, 4000) + 1000 * random.random()
    UE_z = random.uniform(0, 1.5)
    return UE_x, UE_y, UE_z

codebooks = []
codebook_weights_all = dict()
def load_codebooks():
    global codebooks, codebook_weights_all
    # for 4 X 4 antenna size the codebooks
    codebook_files = ["CodeBook1_64Beam.csv", "CodeBook2_64Beam.csv", "CodeBook3_64Beam.csv", "CodeBook4_64Beam.csv",
                      "CodeBook5_64Beam.csv", "CodeBook6_64Beam.csv", "CodeBook7_64Beam.csv", "CodeBook8_64Beam.csv",
                     "CodeBook9_64Beam.csv", "CodeBook10_64Beam.csv", "CodeBook11_64Beam.csv", "CodeBook12_64Beam.csv"]
    #for 16 X 4 antenna size the codebooks
    # codebook_files = ["subarray1_codebook_4X16.csv", "subarray2_codebook_4X16.csv", "subarray3_codebook_4X16.csv", "subarray4_codebook_4X16.csv",
    #                   "subarray5_codebook_4X16.csv", "subarray6_codebook_4X16.csv", "subarray7_codebook_4X16.csv", "subarray8_codebook_4X16.csv",
    #                   "subarray9_codebook_4X16.csv", "subarray10_codebook_4X16.csv", "subarray11_codebook_4X16.csv", "subarray12_codebook_4X16.csv"]

    num_subarrays = len(codebook_files)

    for subarray_index in range(1, num_subarrays + 1):
        try:
            codebook_file = codebook_files[subarray_index - 1]  # Adjust the index by subtracting 1
            codebook = np.loadtxt(codebook_file, delimiter=",", skiprows=1)  # Skip the header row
            codebooks.append(codebook)
        except FileNotFoundError:
            codebooks.append(None)

    codebook_weights_all = {i: load_datasets(codebook) for i in range(len(codebooks))}
    # print(codebook_weights_all)

load_codebooks()

# Define the number of subarrays, sectors, and codebook files
def generate_beam_patterns(base_station_count, num_sectors, num_UEs_per_sector):
    # print('function rN')
    global codebook_weights_all

    Array_AF = dict()

    # Precompute constant values
    frequency = 12e9  # frequency
    lamda = 3e8 / frequency  # wavelength, c = velocity of light = 299792458 m/s ≅ 3 × 10^8 m/s.
    # numel_x = 4  # number of antenna elements along x-axis, M
    numel_x = 4 # number of antenna elements along x-axis, M
    numel_y = 4  # number of antenna elements along y-axis, N
    dx = lamda / 2  # spacing between antenna elements along x-axis
    dy = lamda / 2  # spacing between antenna elements along y-axis

    # Loop over all possible subarrays in each sector
    for subarray in codebook_weights_all:
        Array_AF[subarray] = {}
        # get phase values for all theta phi pairs
        for theta, phi in codebook_weights_all[subarray]:
            phases = codebook_weights_all[subarray][(theta, phi)]

            # Array indexing adjustment for right input to right antenna
            new_order = []
            size = len(phases)
            n = int(size ** 0.5)  # Assuming the size is a perfect square

            for j in range(n):
                for i in range(n):
                    new_order.append(i * n + j)

            # Right input phase to right antenna
            arr = phases.astype(str)
            arr_reordered = [arr[i] for i in new_order]
            phase_degrees = ', '.join(arr_reordered)

            # Convert the phase values to radians
            phase_degrees = [float(value) for value in phase_degrees.split(',') if value]  # Exclude empty strings
            phase_radians = np.deg2rad(phase_degrees)

            # ... Planar Array Factor Computation ...
            AF = np.zeros((360, 360), dtype='complex_')
            r = np.zeros((2, 360*360))
            Pn = np.zeros((numel_x*numel_y, 2))

            for theta in range(0, 360):
                for phi in range(0, 360):
                    r[0, phi+(theta)*360] = np.cos(np.radians(theta)) * np.cos(np.radians(phi))
                    r[1, phi+(theta)*360] = np.sin(np.radians(theta)) * np.sin(np.radians(phi))

            for Pnx in range(0, numel_x):
                for Pny in range(0, numel_y):
                    Pn[Pny+(Pnx)*numel_y, 0] = Pnx*dy
                    Pn[Pny+(Pnx)*numel_y, 1] = Pny*dx

            # Compute the Array Factor for the current beam
            AF = np.matmul(np.exp(1j * phase_radians), np.exp(-1j * (2* np.pi/lamda) * np.matmul(Pn, r)))
            AF = np.reshape(AF, (360, 360))
            for theta in range(360):
                for phi in range(360):
                    Array_AF[subarray][(theta,phi)]= AF[theta,phi]
            # print('eikahne')
            #Array_AF[subarray][(theta,phi)] = AF

    return Array_AF

def generate_beam_scores(base_station_count, UE_X, UE_Y, UE_Z, num_sectors, num_UEs_per_sector):
    print('geRtwe cakked')
    global codebook_weights_all, Array_AF, beam_scores
    # # Define the number of UEs per sector (adjust this number as needed)

    # # Create arrays to store UE positions
    num_UEs_per_BS = num_UEs_per_sector * num_sectors

    # Initialize an array to store the beam scores
    beam_scores = dict()

    # Precompute constant values
    frequency = 12e9  # frequency
    lamda = 3e8 / frequency  # wavelength, c = velocity of light = 299792458 m/s ≅ 3 × 10^8 m/s.
    # numel_x = 4  # number of antenna elements along x-axis, M
    numel_x = 4  # number of antenna elements along x-axis, M
    numel_y = 4  # number of antenna elements along y-axis, N
    dx = lamda / 2  # spacing between antenna elements along x-axis
    dy = lamda / 2  # spacing between antenna elements along y-axis

    # Loop over the 33 base stations
    for k in tqdm(range(base_station_count), desc='Base Stations'):
        beam_scores[k] = {}
        # Loop over all possible subarrays in each sector
        for subarray in codebook_weights_all:
            beam_scores[k][subarray] = {}
            # get phase values for all theta phi pairs
            for theta, phi in codebook_weights_all[subarray]:
                # Identify UEs covered by the main lobe for the current beam
                UEs_main_lobe = 0
                for i in range(num_UEs_per_BS):
                    theta_idx = int((180 / np.pi) * np.arccos(UE_X[k*num_UEs_per_BS+i] / np.sqrt(UE_X[k*num_UEs_per_BS+i]**2 + UE_Y[k*num_UEs_per_BS+i]**2 + UE_Z[k*num_UEs_per_BS+i]**2)))
                    phi_idx = int((180 / np.pi) * np.arctan2(UE_Y[k*num_UEs_per_BS+i], UE_X[k*num_UEs_per_BS+i]))
                    if(phi_idx <0):
                        # print("SAAAAAAAALLLLLLLLLLLLLLAAAAAAAAAAAAAA ", phi_idx)
                        phi_idx = 360 + phi_idx
                    gain = 10 * np.log10(np.abs(Array_AF[subarray][(theta_idx, phi_idx)]))

                    # Set a threshold to distinguish between main lobe and side lobes
                    threshold = 0.5 * np.max(np.abs(Array_AF[subarray][(theta_idx, phi_idx)]))

                    if gain > threshold:
                        UEs_main_lobe += 1

                # Compute the score for the current beam (number of UEs covered by the main lobe)
                score = UEs_main_lobe / num_UEs_per_BS
                beam_scores[k][subarray][(theta,phi)] = score


    return beam_scores

# Call the generate_beam_patterns function
# generate_beam_patterns()
# populate_max_beam_angles()
# import time

def gain_5g(theta, phi, theta_tilt, phi_scan):
    # print(time.time())
    global codebook_weights_all
    beam_pattern, beam_pattern_subarray = beam_pattern_5g(theta, phi, theta_tilt, phi_scan)
    # print(beam_pattern_subarray)
    gain_subarray = {}
    if beam_pattern == 0:
        gain = gain_antenna_element(theta, phi)
        # print(codebook_weights_all[1].keys())
        gain_subarray = {subarray:
                             {(theta, phi): gain_antenna_element(theta, phi) for theta, phi in beam_pattern_subarray[subarray]}
                         for subarray in beam_pattern_subarray}

        # for subarray in beam_pattern_subarray:
        #     gain_subarray[subarray] = {}
        #     for theta, phi in beam_pattern_subarray[subarray]:
        #         gain_subarray[subarray][theta, phi)] = gain_antenna_element(theta, phi)
        #



    else:
        # print('anything', beam_pattern, math.log10(beam_pattern), theta, phi)
        # try:
        gain = gain_antenna_element(theta, phi) + 10 * math.log10(abs(beam_pattern))
        # except:
        #     print(beam_pattern)
        gain_subarray = {subarray:
                             {(theta, phi): gain_antenna_element(theta, phi) + 10 * math.log10(abs(
                                 beam_pattern_subarray[subarray][(int(theta),int(phi))])) for theta, phi in beam_pattern_subarray[subarray]}
                         for subarray in beam_pattern_subarray}


    # print(time.time())
    return gain, gain_subarray




# space refers to D/λ
# def get_phi_min(space) -> float:
#     # Rec. ITU-R S.465-6
#     if space >= 50:
#         return max(1.0, 100 * (1 / space))
#     else:
#         return max(2.0, 114 * (space ** -1.09))


# receiving antenna gain of FSS earth station
# phi is an angle between base station antenna direction and FSS ES antenna's main axis (elevation angle)
def gain_fss_s1428(phi, phi_min) -> float:
    if 0 < phi < phi_min:
        return 32 - 25 * math.log10(phi_min)
    elif phi_min <= phi < 48:
        return 32 - 25 * math.log10(phi)
    elif 48 <= phi <= 180:
        return -10

    # This function is only defined where phi is within (0, 180]
    raise ValueError(
        f"Angle phi must be within the interval (0, 180] degrees, was {phi} degrees instead"
    )


# co-polarized components only
# https://www.etsi.org/deliver/etsi_en/303900_303999/303981/01.02.01_60/en_303981v010201p.pdf
def gain_fss_wbes_b(phi) -> float:
    if 0 <= phi < 6:
        return 20
    elif 6 <= phi < 48:
        return 40 - 25 * math.log10(phi)
    elif 48 <= phi <= 180:
        return -2
    elif 180 <= phi <= 360:
        return gain_fss_wbes_b(360 - phi)

    # This function is only defined where phi is within [6, 180]
    raise ValueError(
        f"Angle phi must be within the interval [6, 180] degrees, was {phi} degrees instead"
    )


class Building:
    def __init__(self, coordinates=None, height=None, lat_FSS=None, lon_FSS=None):
        self.coordinates = []
        if coordinates is not None:
            self.coordinates = coordinates
        if height is None or not str(height).isdigit():
            self.height = random.uniform(10, 40)
        else:
            self.height = height

        if lat_FSS is None:
            self.lat_FSS = 37.20250
        else:
            self.lat_FSS = lat_FSS

        if lon_FSS is None:
            self.lon_FSS = -80.43444
        else:
            self.lon_FSS = lon_FSS

        self.x_coord, self.y_coord, self.z_coord = self.latlon_to_XYZ(lat_FSS, lon_FSS)

        self.points = []
        for i in range(len(self.x_coord)):
            x, y, z = self.x_coord[i], self.y_coord[i], self.z_coord[i]
            self.points.append(Point(x, y, z))

        self.xy_points = np.array([(p.x, p.y) for p in self.points])
        self.xy_polygon = geometry.Polygon(self.xy_points)

        self.wall_polygons = self.get_wall_polygons()

    def latlon_to_XYZ(self, lat_FSS, lon_FSS):
        x_coord = np.array([])
        y_coord = np.array([])
        z_coord = np.array([])

        # lat_FSS = 37.20250
        # lon_FSS = -80.43444
        R = 6.371e6  # Radius of the earth

        x_FSS = R * math.cos(math.radians(lat_FSS)) * math.cos(math.radians(lon_FSS))
        y_FSS = R * math.cos(math.radians(lat_FSS)) * math.sin(math.radians(lon_FSS))

        # bottom points of building at each coordinate
        for lon, lat in self.coordinates:
            x = R * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
            y = R * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
            x_coord = np.append(x_coord, x)
            y_coord = np.append(y_coord, y)
            z_coord = np.append(z_coord, 0)
            # z_coord = np.append(z_coord, self.height)

        # make coordinates to relative to FSS position
        x_coord -= x_FSS
        y_coord -= y_FSS

        return x_coord, y_coord, z_coord

    def get_wall_polygons(self):
        polygons = []

        # points = [Point(x,y,0) for x,y in p.boundary.coords]
        # polygons.append(ConvexPolygon(points))
        points = [Point(x, y, self.height) for x, y in self.xy_polygon.boundary.coords]
        # polygons.append(ConvexPolygon(points))

        for i in range(len(points) - 1):
            high_point_1 = points[i]
            high_point_2 = points[i + 1]
            low_point_1 = Point(high_point_1.x, high_point_1.y, 0)
            low_point_2 = Point(high_point_2.x, high_point_2.y, 0)

            poly_points = [low_point_1, low_point_2, high_point_1, high_point_2]
            # print(poly_points)
            polygons.append(ConvexPolygon(poly_points))

        return polygons


class FSS_Channels:
    # channels start at 0, so a channel plan with 8 channels is from 0 to 7
    # RR - these variables needed as input
    band_width = 240e6  # 240MHz

    # 10.7-12.7 GHz frequency range
    range_start = 10.7e9  # 10.7GHz
    range_end = 12.7e9  # 12.7GHz

    channel_count = 8

    range_size = range_end - range_start
    unused_space = range_size - band_width * channel_count

    # FSS channel spacing should be 10MHz
    space_between_channels = unused_space / channel_count

    @staticmethod
    def getChannelRange(channel) -> Tuple[int, int]:
        channel_start = int(
            (0.5 + channel) * FSS_Channels.space_between_channels
            + channel * FSS_Channels.band_width
        )
        channel_end = channel_start + FSS_Channels.band_width
        return channel_start, channel_end

    def __init__(self, channel):
        self.channel = channel


class BS_Channels:
    # channels start at 1, so a channel plan with 5 channels is from 1 to 5

    # RR - these variables needed as input
    band_width = 100e6  # 100MHz

    # 12.2-12.7 GHz frequency range
    range_start = 12.2e9  # 12.2GHz
    range_end = 12.7e9  # 12.7GHz

    channel_count = 5

    range_size = range_end - range_start
    unused_space = range_size - band_width * channel_count

    # BS channel spacing should be 0MHz
    space_between_channels = unused_space / channel_count

    @staticmethod
    def getChannelRange(channel) -> Tuple[int, int]:
        channel -= 1

        channel_start = int(
            (0.5 + channel) * BS_Channels.space_between_channels
            + channel * BS_Channels.band_width
        )
        channel_end = channel_start + BS_Channels.band_width
        return channel_start, channel_end

    def __init__(self, channel):
        self.channel = channel


# In[ ]:
class Context:
    pass


class BS:
    def __init__(self, radius, max_height, carr_freq, interference_type):
        self.radius = radius
        self.max_height = max_height
        self.carr_freq = carr_freq
        self.interference_type = interference_type
        # RR - these variables needed as input
        self.base_heights = {
            "UMi": 10,
            "UMa": 25,
            "RMa": 35,
            None: random.randint(10, 35),
        }
        self.BS_x = random.randint(-4000, 4000) + 1000 * random.random()
        self.BS_y = random.randint(-4000, 4000) + 1000 * random.random()
        self.BS_z = self.base_heights[self.interference_type]

    def BS_random_co_ordinates(self):
        return self.BS_x, self.BS_y, self.BS_z

    def UE_random_co_ordinates(self):
        UE_x = random.randint(-4000, 4000) + 1000 * random.random()
        UE_y = random.randint(-4000, 4000) + 1000 * random.random()
        UE_z = random.uniform(0, 1.5)

        return UE_x, UE_y, UE_z

    # Creating the FSS class
    def FSS_random_co_ordinates(self):
        FSS_x = random.randint(-4000, 4000) + 1000 * random.random()
        FSS_y = random.randint(-4000, 4000) + 1000 * random.random()
        FSS_z = random.choice([1.5, 4.5])

        return FSS_x, FSS_y, FSS_z


app = Flask(__name__)


@app.route('/parsesimulatordata', methods=['POST'])
def parse_simulator_data():
    # Get the input data from the DSA framework
    json_data = request.get_json()

    # Parse the JSON data to variables
    lat_FSS = json_data['lat_FSS']
    lon_FSS = json_data['lon_FSS']
    radius = json_data['radius']
    simulation_count = json_data['simulation_count']
    bs_ue_max_radius = json_data['bs_ue_max_radius']
    bs_ue_min_radius = json_data['bs_ue_min_radius']
    base_station_count = json_data['base_station_count']
    rain = json_data['rain']
    rain_rate = json_data['rain_rate']
    exclusion_zone_radius = json_data['exclusion_zone_radius']
    # Parse the base station data into a list of dictionaries
    base_stations = []
    for bs_data in json_data['base_stations']:
        base_station = {
            'cid': bs_data['cid'],
            'latitude': bs_data['latitude'],
            'longitude': bs_data['longitude'],
            'range': bs_data['range'],
            'samples': bs_data['samples'],
            'averageSignal': bs_data['averageSignal'],
            'changeable': bs_data['changeable'],
            'lac': bs_data['lac'],
            'mcc': bs_data['mcc'],
            'mnc': bs_data['mnc'],
            'radio': bs_data['radio'],
            'status': bs_data['status'],
            'unique_id': bs_data['unique_id'],
            'unit': bs_data['unit'],
            'updated': bs_data['updated'],
            'dist_from_FSS': bs_data['dist_from_FSS']
        }
        base_stations.append(base_station)


    # Run the simulator with the parsed data
    output_data = run_simulator(lat_FSS, lon_FSS, radius, simulation_count, bs_ue_max_radius, bs_ue_min_radius,
                                base_station_count, rain, rain_rate, exclusion_zone_radius, base_stations)

    # output_data = {
    #     "Interference_values_UMi_each_Bs": [
    #         -31.586625574403882,
    #         -35.191424022093855,
    #         -18.56990745856593,
    #         -35.47686461913297,
    #         -38.474774043668745,
    #         -11.141635807315321,
    #         -62.20374665019345,
    #         -19.370609097823518,
    #         -26.541763404108263,
    #         -47.50175471126143,
    #         -17.88587437929757,
    #         -35.195019551854266,
    #         -52.674793302083586,
    #         -29.62489982828368,
    #         2.7971202581528343,
    #         6.107579300585324,
    #         -25.10698899447582,
    #         -69.47948320793697,
    #         -50.725313822900524,
    #         -55.20441399983092,
    #         -31.93346957378436,
    #         -40.95931464486423,
    #         -39.14056440921604,
    #         -21.701477074230823,
    #         -32.91510580409362,
    #         -31.42415785610693,
    #         -35.99502355385156,
    #         -26.626390561350718,
    #         -25.523930166508602,
    #         -23.923752659832083,
    #         -41.86998035329006,
    #         -47.895659185236084,
    #         -37.21194865204484
    #     ]
    # }

    # Return the output as a JSON response
    return jsonify(output_data)


def get_plot():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    # Encode the contents of the buffer as a base64 string
    buffer.seek(0)
    contents = buffer.getvalue()
    encoded = base64.b64encode(contents).decode('utf-8')
    # Embed the base64-encoded string in an HTML image tag
    image_html = f'<img src="data:image/png;base64,{encoded}">'
    return image_html


random.seed(10)


def run_simulator(lat_FSS, lon_FSS, radius, simulation_count, bs_ue_max_radius, bs_ue_min_radius, base_station_count,
                  rain, rain_rate, exclusion_zone_radius, base_stations):
    global Array_AF, beam_scores
    simulator_result = {}
    # Structure: (theta, phi) -> (theta_etilt, phi_scan)
    saved_tp = dict()
    saved_tp_file = Path("t0p0.pkl")
    ctx = Context()
    ctx.rain = rain
    ctx.rain_rate = rain_rate
    ctx.lat_FSS = lat_FSS
    ctx.lon_FSS = lon_FSS
    if saved_tp_file.is_file():
        with open(saved_tp_file, "rb") as f:
            saved_tp = pickle.load(f)
    # Structure: (segment, polygon) -> (boolean True or False)
    saved_los = dict()
    saved_los_file = Path("los.pkl")
    if saved_los_file.is_file():
        with open(saved_los_file, "rb") as f:
            saved_los = pickle.load(f)

    # saved_beam_scores_file = open("beam_scores.pkl", "rb")
    # beam_scores = pickle.load(saved_beam_scores_file)


    beam_scores = []
    saved_beam_scores_file = Path("beam_scores.pkl")
    if saved_beam_scores_file.is_file():
        with open(saved_beam_scores_file, "rb") as f:
            beam_scores = pickle.load(f)
    ctx.beam_scores = beam_scores
    if ctx.beam_scores is None:
        exit()
    Array_AF = []
    saved_array_af_file = Path("array_af.pkl")
    if saved_array_af_file.is_file():
        with open(saved_array_af_file, "rb") as f:
            Array_AF = pickle.load(f)

    data_within_zone = pd.DataFrame(base_stations)
    R = 6.371e6  # Radius of the earth

    x_FSS = R * math.cos(math.radians(lat_FSS)) * math.cos(math.radians(lon_FSS))
    y_FSS = R * math.cos(math.radians(lat_FSS)) * math.sin(math.radians(lon_FSS))
    z_FSS = R * math.sin(math.radians(lat_FSS))

    ctx.x_FSS = x_FSS
    ctx.y_FSS = y_FSS
    ctx.z_FSS = z_FSS

    print("X, Y, Z = " + str([x_FSS, y_FSS, z_FSS]))
    # radius = 5000  # radius of the inclusion zone

    # latitude_range = radius / 110574
    # longitude_range = radius / (111320 * math.cos(math.radians(latitude_range)))
    # print("Total area of inclusion zone is = " + str(radius * radius * math.pi))
    # print(latitude_range)
    # print(longitude_range)
    # data_within_zone = data[
    #     (data["radio"] == "GSM")
    #     & (data["longitude"] <= (lon_FSS + longitude_range))
    #     & (data["longitude"] >= (lon_FSS - longitude_range))
    #     & (data["latitude"] <= (lat_FSS + latitude_range))
    #     & (data["latitude"] >= (lat_FSS - latitude_range))
    #     ]
    #
    # data_within_zone.head(10)
    # len(data_within_zone)
    with open("data/export (1).geojson") as f:
        df = json.load(f)
        data1 = pd.json_normalize(df, record_path=["features"])
    # data1.head(10)
    # data1[data1["properties.height"].notnull()].head(20)
    # data1["geometry.coordinates"].head(10)

    # for demo we are not showing the polygon
    # b = Building(
    #     data1.iloc[500]["geometry.coordinates"][0], data1.iloc[500]["properties.height"], lat_FSS, lon_FSS
    # )
    # x, y = b.xy_polygon.boundary.xy
    # plt.plot(x, y)
    # html1 = get_plot()
    # simulator_result["html_polygon_2D"] = html1

    # we are not showing the 3D view in the demo
    # r = Renderer(backend="matplotlib")
    # for p in b.wall_polygons:
    #     r.add((p, "b", 1), normal_length=0)
    # r.show()
    # # https://htmlcodeeditor.com/
    # htmlpolygon = get_plot()
    # simulator_result["html_polygon_3D"] = htmlpolygon

    # buildings = []
    # for i in tqdm(range(len(data1))):
    #     for coords in data1.iloc[i]["geometry.coordinates"]:
    #         try:
    #             buildings.append(Building(coords, data1.iloc[i]["properties.height"], lat_FSS, lon_FSS))
    #             print(f"created building {i}")
    #         except:
    #             print(f"Skipping building {i}")

    buildings = []
    if os.path.isfile("buildings.pkl"):
        with open('buildings.pkl', 'rb') as file:
            buildings = pickle.load(file)
    else:
        for i in tqdm(range(len(data1))):
            for coords in data1.iloc[i]["geometry.coordinates"]:
                try:
                    buildings.append(Building(coords, data1.iloc[i]["properties.height"], lat_FSS, lon_FSS))
                    print(f"created building {i}")
                except:
                    print(f"Skipping building {i}")
        with open("buildings.pkl", "wb") as file:
            pickle.dump(buildings, file)

    ctx.buildings = buildings
    ctx.saved_los = saved_los
    # ctx.beam_scores = beam_scores
    ctx.radius = radius
    ctx.R = R
    k = 1.38064852 * 10 ** (-23)
    T = 200
    B = 240e6
    Noise = 10 * math.log10(k * T * B)
    Noise_W = 10 ** (Noise / 10)
    print("Noise:", Noise)
    print("Noise in Watts:", Noise_W)

    ctx.base_station_count = base_station_count
    ctx.bs_ue_min_radius = bs_ue_min_radius
    ctx.bs_ue_max_radius = bs_ue_max_radius
    ctx.Noise_W = Noise_W
    x, y, z = 0, 0, 4.5
    ctx.x = x
    ctx.y = y
    ctx.z = z
    ctx.data_within_zone = data_within_zone
    ctx.saved_tp = saved_tp
    FSS_phi = {"UMi": 15, "UMa": 48, "RMa": 5}
    ctx.FSS_phi = FSS_phi
    # Prototype functions to calculate antenna gain of 5G base station and FSS earth station
    # https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/14.00.00_60/tr_138901v140000p.pdf
    # values recommended by Dr. Zoheb
    # constant values based on ECC Rep 281

    # antenna horizontal pattern

    (
        distance_RMa,
        I_N_RMa_W,
        I_N_RMa_noAverage,
        distance_UMa,
        I_N_UMa_W,
        I_N_UMa_noAverage,
        distance_UMi,
        I_N_UMi_W,
        I_N_UMi_noAverage,
        line_of_sight,
    ) = [np.empty([0]) for x in range(10)]
    # simulation_count = 1
    for i in tqdm(range(simulation_count)):
        (
            distance_RMa_single,
            I_N_RMa_single_W,
            distance_UMa_single,
            I_N_UMa_single_W,
            distance_UMi_single,
            I_N_UMi_single_W,
            line_of_sight_single,
            saved_los,
        ) = simulate(output=False, ctx=ctx)
        print(f"The current simulation is {i} out of total {simulation_count}")

        distance_RMa = np.append(distance_RMa, distance_RMa_single)
        # print(I_N_RMa_single_W)
        I_N_RMa_W = np.append(I_N_RMa_W, I_N_RMa_single_W)
        I_N_RMa_noAverage = np.append(I_N_RMa_noAverage, 10 * np.log10(I_N_RMa_single_W))

        distance_UMa = np.append(distance_UMa, distance_UMa_single)
        I_N_UMa_W = np.append(I_N_UMa_W, I_N_UMa_single_W)
        I_N_UMa_noAverage = np.append(I_N_UMa_noAverage, 10 * np.log10(I_N_UMa_single_W))

        distance_UMi = np.append(distance_UMi, distance_UMi_single)
        I_N_UMi_W = np.append(I_N_UMi_W, I_N_UMi_single_W)
        I_N_UMi_noAverage = np.append(I_N_UMi_noAverage, 10 * np.log10(I_N_UMi_single_W))

        line_of_sight = np.append(line_of_sight, line_of_sight_single)

    for arr in (I_N_RMa_noAverage, I_N_UMa_noAverage, I_N_UMi_noAverage):
        arr[arr == -np.inf] = 0

    I_N_RMa = 10 * np.log10(np.average(I_N_RMa_W))
    I_N_UMa = 10 * np.log10(np.average(I_N_UMa_W))
    I_N_UMi = 10 * np.log10(np.average(I_N_UMi_W))

    pairs = {
        'RMa': (np.average(distance_RMa), I_N_RMa),
        'UMa': (np.average(distance_UMa), I_N_UMa),
        'UMi': (np.average(distance_UMi), I_N_UMi),
    }

    pairs_noAverage = {
        'RMa': (distance_RMa, I_N_RMa_noAverage),
        'UMa': (distance_UMa, I_N_UMa_noAverage),
        'UMi': (distance_UMi, I_N_UMi_noAverage),
    }

    simulator_result["Interference_values_UMi_each_Bs"] = I_N_UMi_noAverage.tolist()
    # simulator_result["Interference_values_UMi"] = I_N_UMi_W.tolist()

    # TODO NEED TO RECHECK THE VALUES
    with open(saved_tp_file, "wb") as f:
        pickle.dump(saved_tp, f)
    with open(saved_los_file, "wb") as f:
        pickle.dump(saved_los, f)

    # len(pairs_noAverage["RMa"][0])
    #
    # len(pairs_noAverage["RMa"][1])



    box_dict_UMi = dict()
    distance, interface_Noise = pairs_noAverage["UMi"]
    for i in range(len(interface_Noise)):
        if distance[i] not in box_dict_UMi:
            box_dict_UMi[distance[i]] = []
        box_dict_UMi[distance[i]].append([interface_Noise[i], line_of_sight[i]])

    # with open('temp\\data.pkl', 'wb') as f:
    #     pickle.dump(box_dict_UMi, f)

    fig, ax = plt.subplots()
    # Creating plot
    keys = sorted([key for key in box_dict_UMi])
    threshold = -8.5
    if rain:
        threshold = -12

    x_axis, y_axis, colour = [], [], []
    for i, key in enumerate(keys):

        INR_val_bs = box_dict_UMi[key][0][0]
        line_of_sight1 = box_dict_UMi[key][0][1]
        x_axis.append(key)
        y_axis.append(INR_val_bs)
        if line_of_sight1 == 1.0:
            color = 'blue'
        else:
            color = 'red'
        colour.append(color)

    plt.scatter(x_axis, y_axis, c=colour)

    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', linestyle='--', lw=2),
                    Line2D([0], [0], color='black', linestyle='--', lw=2)]
    ax.legend(custom_lines,
              ['LOS', 'NLOS', 'Exclusion Zone ({}m)'.format(exclusion_zone_radius),
               'Threshold ({}dB)'.format(threshold)],
              fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    # ax.set_xticklabels([int(key) if not i % 4 else "" for i, key in enumerate(keys)], fontsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_xlabel('Distance of Each BS From FSS (meters)', fontsize=10)
    plt.axhline(y=threshold, color='black', linestyle='--', label='Threshold {}'.format(threshold))
    plt.axvline(x=exclusion_zone_radius, color='green',
                linestyle='--', label='Exclusion Zone {}'.format(exclusion_zone_radius))
    ax.set_ylabel('I/N (dB)', fontsize=10)
    fig.set_size_inches(12, 4)
    plt.tight_layout()

    # show plot
    html12 = get_plot()
    simulator_result["html_Interference_Noise"] = html12
    # TODO need to add a horizontal and vertical line for (Exz and I/N threshold)
    # TODO can use the distance from the dataset no need to calculate

    # # plt.show()
    # Elevation angles (FSS towards to sky )
    # fig, ax = plt.subplots()
    # FSS_phi = {"UMi": 15, "UMa": 48, "RMa": 5}
    # contexts = sorted([context for context in FSS_phi], key=lambda x: FSS_phi[x])
    # boxplots = []
    # boxplots = ax.boxplot([pairs_noAverage[context][1] for context in contexts])
    # ax.set_xticklabels([FSS_phi[context] for context in contexts])
    # ax.set_title("I/N Vs Elevation graph")
    # ax.set_xlabel("Elevation (degrees)")
    # ax.set_ylabel("I/N (db)")
    # ax.legend()
    # html13 = get_plot()
    # plt.show()
    # line_of_sight
    # Creating dataset
    # box_dict_UMi = dict()
    # distance, interface_Noise = pairs_noAverage["UMi"]
    # for i in range(len(interface_Noise)):
    #     if distance[i] not in box_dict_UMi:
    #         box_dict_UMi[distance[i]] = []
    #     box_dict_UMi[distance[i]].append([interface_Noise[i], line_of_sight[i]])
    #
    # # print(box_dict_UMi)
    #
    # fig, ax = plt.subplots()
    #
    # # Creating plot
    # keys = sorted([key for key in box_dict_UMi])
    # for i, key in enumerate(keys):
    #     # key is distance
    #     # print(key)
    #     # print(box_dict_UMi[key])
    #     line_of_sight1 = box_dict_UMi[key][0][1]
    #     ax.boxplot(
    #         [I_N for I_N, los in box_dict_UMi[key]],
    #         patch_artist=True,
    #         positions=[i],
    #         boxprops=dict(facecolor="white", color="blue" if line_of_sight1 else "red"),
    #     )
    #
    # # keys = sorted([key for key in box_dict_UMi])
    # # ax.boxplot([box_dict_UMi[key] for key in keys])
    # ax.set_xticklabels([int(i) for i in keys])
    #
    # ax.set_title("I/N Vs Distance graph (UMi)")
    # ax.set_xlabel("Distnace of Each BS to FSS (meters)")
    # ax.set_ylabel("I/N (db)")
    # ax.legend()
    # # show plot
    # # plt.show()
    # fig.set_size_inches(18, 8.5)
    # # fig.savefig('graphUmi.pdf', dpi=100)
    # html14 = get_plot()
    # # plt.show()
    # # Creating dataset
    # box_dict_UMa = dict()
    # distance, interface_Noise = pairs_noAverage["UMa"]
    # for i in range(len(interface_Noise)):
    #     if distance[i] not in box_dict_UMa:
    #         box_dict_UMa[distance[i]] = []
    #     box_dict_UMa[distance[i]].append([interface_Noise[i], line_of_sight[i]])

    # print(box_dict_UMi)

    # fig, ax = plt.subplots()
    #
    # # Creating plot
    # keys = sorted([key for key in box_dict_UMa])
    # for i, key in enumerate(keys):
    #     # key is distance
    #     # print(key)
    #     # print(box_dict_UMi[key])
    #     line_of_sight1 = box_dict_UMa[key][0][1]
    #     ax.boxplot(
    #         [I_N for I_N, los in box_dict_UMa[key]],
    #         patch_artist=True,
    #         positions=[i],
    #         boxprops=dict(facecolor="white", color="blue" if line_of_sight1 else "red"),
    #     )
    #
    # # keys = sorted([key for key in box_dict_UMi])
    # # ax.boxplot([box_dict_UMi[key] for key in keys])
    # ax.set_xticklabels([int(i) for i in keys])
    #
    # ax.set_title("I/N Vs Distance graph (UMa)")
    # ax.set_xlabel("Distnace of Each BS to FSS (meters)")
    # ax.set_ylabel("I/N (db)")
    # ax.legend()

    # show plot
    # plt.show()
    # fig.set_size_inches(18, 8.5)
    # # fig.savefig('graphUMa.pdf', dpi=100)
    # html15 = get_plot()
    #
    # # plt.show()
    # # Creating dataset
    # box_dict_RMa = dict()
    # distance, interface_Noise = pairs_noAverage["RMa"]
    # for i in range(len(interface_Noise)):
    #     if distance[i] not in box_dict_RMa:
    #         box_dict_RMa[distance[i]] = []
    #     box_dict_RMa[distance[i]].append([interface_Noise[i], line_of_sight[i]])
    #
    # # print(box_dict_UMi)
    #
    # fig, ax = plt.subplots()
    #
    # # Creating plot
    # keys = sorted([key for key in box_dict_RMa])
    # for i, key in enumerate(keys):
    #     # key is distance
    #     # print(key)
    #     # print(box_dict_UMi[key])
    #     line_of_sight1 = box_dict_RMa[key][0][1]
    #     ax.boxplot(
    #         [I_N for I_N, los in box_dict_RMa[key]],
    #         patch_artist=True,
    #         positions=[i],
    #         boxprops=dict(facecolor="white", color="blue" if line_of_sight1 else "red"),
    #     )
    #
    # ax.set_xticklabels([int(i) for i in keys])
    # ax.set_title("I/N Vs Distance graph (RMa)")
    # ax.set_xlabel("Distnace of Each BS to FSS (meters)")
    # ax.set_ylabel("I/N (db)")
    # ax.legend()
    # # show plot
    # # plt.show()
    # fig.set_size_inches(18, 8.5)
    # # fig.savefig('graphRma.pdf', dpi=100)
    # html16 = get_plot()
    # # plt.show()
    # return output
    #
    return simulator_result


if __name__ == '__main__':
    # Use a self-signed certificate for testing purposes
    context = ('cert.pem', 'key.pem')
    app.run(debug=True, ssl_context=context)
