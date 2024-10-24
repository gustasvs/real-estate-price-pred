"use client";

import React, { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { updateUserProfile as updateUserProfileApi } from "../../actions/user";
import { Input, Button, Select, message, Upload, Spin, Divider } from "antd";
import { PageHeader } from "@ant-design/pro-components";
import { UploadOutlined } from "@ant-design/icons";
import styles from "./Profile.module.css"; // Assuming you have CSS module for styling

const UserProfilePage = () => {
  const { data: session, status } = useSession();
  const router = useRouter();

  const [profile, setProfile] = useState({
    name: "",
    email: "",
    role: "ADMIN", // Default role
    image: "",
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (session) {
      setProfile({
        name: session.user?.name || "",
        email: session.user?.email || "",
        role: session.user?.role || "ADMIN",
        image: session.user?.image || "",
      });
    }
  }, [session]);

  const handleUpdateProfile = async () => {
    if (!profile.name.trim() || !profile.email.trim()) {
      message.error("Vārds un e-pasts nevar būt tukši");
      return;
    }

    setSaving(true);
    try {
      await updateUserProfileApi({
        name: profile.name,
        email: profile.email,
        role: profile.role,
        image: profile.image,
      });
      message.success("Profils veiksmīgi atjaunināts!");
    } catch (error) {
      console.error("Kļūda atjauninot profilu:", error);
      message.error("Kļūda atjauninot profilu.");
    } finally {
      setSaving(false);
    }
  };

  const handleImageUpload = (file) => {
    // Implement your logic to upload the image (e.g., upload to a server or cloud storage)
    // After uploading, set the image URL in the profile state
    const uploadedUrl = URL.createObjectURL(file); // This is just for demo purposes
    setProfile((prev) => ({ ...prev, image: uploadedUrl }));
    return false; // Prevents the default upload behavior
  };

  if (status === "loading") {
    return (
      <div className={styles["loading-container"]}>
        <Spin size="large" />
      </div>
    );
  }

  if (!session) {
    router.push("/login");
    return null;
  }

  return (
    <div className={styles["profile-page-container"]}>
      <PageHeader title="Mans Profils" />
      <Divider />

      <div className={styles["form-container"]}>
        <label className={styles["form-label"]}>Vārds</label>
        <Input
          value={profile.name}
          onChange={(e) => setProfile({ ...profile, name: e.target.value })}
          placeholder="Name"
        />

        <label className={styles["form-label"]} style={{ marginTop: "1rem" }}>
          E-pasts
        </label>
        <Input
          value={profile.email}
          onChange={(e) => setProfile({ ...profile, email: e.target.value })}
          placeholder="Email"
        />

        <label className={styles["form-label"]} style={{ marginTop: "1rem" }}>
          Loma
        </label>
        <Select
          value={profile.role}
          onChange={(value) => setProfile({ ...profile, role: value })}
          options={[
            { label: "Admins", value: "ADMIN" },
            { label: "Lietotājs", value: "USER" },
          ]}
        />

        <label className={styles["form-label"]} style={{ marginTop: "1rem" }}>
          Profila bilde
        </label>
        <Upload
          beforeUpload={handleImageUpload}
          showUploadList={false}
          accept="image/*"
        >
          <Button icon={<UploadOutlined />}>Augšupielādēt bildi</Button>
        </Upload>
        {profile.image && (
          <img
            src={profile.image}
            alt="Profile"
            className={styles["profile-image-preview"]}
            style={{ marginTop: "1rem" }}
          />
        )}

        <Button
          type="primary"
          onClick={handleUpdateProfile}
          loading={saving}
          style={{ marginTop: "1.5rem" }}
        >
          Saglabāt izmaiņas
        </Button>
      </div>
    </div>
  );
};

export default UserProfilePage;
