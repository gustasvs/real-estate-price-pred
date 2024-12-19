import React, { Suspense } from 'react';

// Define a fallback component, which can be a simple loader or a more complex skeleton
const FallbackLoading = () => (
  <div>Loading...</div> // Replace with your preferred loading indicator or skeleton
);

// Define the Loading component
const Loading = () => {
  return (
    <>
    {/* <Suspense fallback={<FallbackLoading />}>
      Components that you want to load lazily
    </Suspense> */}
    </>
  );
};

export default Loading;
