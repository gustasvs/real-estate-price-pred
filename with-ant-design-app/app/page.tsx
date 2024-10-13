"use client";

import React, { useEffect, useState } from "react";
import Banner from "./components/landing_page/banner";
import Statistics from "./components/landing_page/stats";
import GenericLayout from "./components/generic-page-layout";

const HomePage = () => {
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => {
    setHydrated(true);
  }, []);

  return (
    
      <GenericLayout homePage>
        {/* Conditionally render Banner and Statistics after hydration */}
        {hydrated && (
          <>
            <Banner />
            <Statistics />
          </>
        )}
      </GenericLayout>
  );
};

export default HomePage;
