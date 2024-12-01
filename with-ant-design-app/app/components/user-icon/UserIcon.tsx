"use client";

import React, { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { Session } from "next-auth";

import styles from "./UserIcon.module.css";
import { generateDownloadUrl } from "../../api/generateDownloadUrl";

interface UserIconProps {
  onClick?: () => void;
}

const UserIcon: React.FC<UserIconProps> = ({ onClick }) => {
  const { data: session, status } = useSession() as {
    data: Session | null;
    status: string;
  };

  const [userImageUrl, setUserImageUrl] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserImage = async () => {
      if (session?.user?.image) {
        const sessionUserImage = session.user.image;
        const downloadUrl = await generateDownloadUrl(sessionUserImage, "profile-pictures");

        // console.log("Download URL:", downloadUrl);

        if (typeof downloadUrl === "object" && "error" in downloadUrl) {
          console.error("Error getting user image URL:", downloadUrl.error);
        } else {
          setUserImageUrl(downloadUrl);
        }
      }
    };

    fetchUserImage();
  }, [session]);


  // console.log("User Image URL:", userImageUrl);

  return (
    <div className={styles.container} onClick={onClick}>
      {userImageUrl ? (
        <img
          src={userImageUrl}
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
