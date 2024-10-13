


import { redirect } from "next/navigation";
import React from "react";
import { auth } from "../../auth";

const Server = async () => {
  const session = await auth();
  console.log(session, "SERVER PAGE");
  if (!session?.user) {
    redirect("/");
  }
  return (
    <main className="flex h-full items-center justify-center flex-col gap-2">
      <h1 className="text-3xl">Server page</h1>
      <p className="text-lg">{session?.user?.email}</p>
    </main>
  );
};

export default Server;
