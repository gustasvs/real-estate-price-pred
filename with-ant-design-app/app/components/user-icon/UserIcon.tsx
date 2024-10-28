"use client";

import React from "react";
import { useSession } from "next-auth/react";
import { Session } from "next-auth";

import styles from "./UserIcon.module.css";

interface UserIconProps {
  onClick?: () => void;
}

const UserIcon: React.FC<UserIconProps> = ({ onClick }) => {
  const { data: session, status } = useSession() as {
    data: Session | null;
    status: string;
  };

  return (
    <div className={styles.container} onClick={onClick}>
      {session?.user?.image ? (
        <img
          src={session?.user?.image}
          alt="User Image"
          className={styles["user-image"]}
        />
      ) : (
        <div className={styles["blank-user-image"]} />
      )}
    </div>
  );
};

export default UserIcon;
