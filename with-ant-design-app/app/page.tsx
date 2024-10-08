"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { SmileFilled } from "@ant-design/icons";
import {
  Button,
  DatePicker,
  Form,
  InputNumber,
  Select,
  Slider,
  Switch,
  ConfigProvider,
} from "antd";
import theme from "./themeConfig";
import Banner from "./components/landing_page/banner";
import Statistics from "./components/landing_page/stats";
import Navbar from "./components/navigation/navbar";
import GenericLayout from "./components/generic-page-layout";
import { initParticlesEngine } from "@tsparticles/react";



const HomePage = () => {
  return (  
  <ConfigProvider theme={theme}>
    <GenericLayout homePage>
    <Banner />
    <Statistics />
    </GenericLayout>
  </ConfigProvider>
  );
};

export default HomePage;
