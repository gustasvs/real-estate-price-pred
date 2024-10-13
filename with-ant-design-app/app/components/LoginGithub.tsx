"use client";
import { GithubFilled } from "@ant-design/icons";

import React from "react";
import { login } from "../../actions/auth";

const LoginGithub = () => {
  return (
    <div
      onClick={() => login("github")}
      className="w-full gap-4  hover:cursor-pointer mt-6 h-12 bg-black rounded-md p-4 flex justify-center items-center"
    >
      <GithubFilled/>
      <p className="text-white">Login with Github</p>
    </div>
  );
};

export default LoginGithub;
