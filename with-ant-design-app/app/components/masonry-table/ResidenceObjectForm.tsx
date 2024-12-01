"use client";

import React, { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import {
  Button,
  Form,
  Input,
  Upload,
  message,
  UploadProps,
  InputNumber,
  Row,
  Col,
} from "antd";
import { InboxOutlined, UploadOutlined } from "@ant-design/icons";

import imageCompression from "browser-image-compression";

import styles from "./ResidenceObjectForm.module.css";
import {
  createObject as createObjectApi,
  getObject as getObjectApi,
  updateObject as updateObjectApi,
} from "../../../actions/groupObjects";
import { StyledNumberInput, StyledSwitch, StyledTextField } from "../my-profile/my-profile-form/MyProfileForm";
import { generateUploadUrl } from "../../api/generateUploadUrl";
import { useSession } from "next-auth/react";

const { Dragger } = Upload;

const props: UploadProps = {
  name: "file",
  multiple: true,
  listType: "picture",
  // action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
  onChange(info) {
    const { status } = info.file;
    if (status !== "uploading") {
      console.log(info.file, info.fileList);
    }
    if (status === "done") {
      message.success(`${info.file.name} fails augšupielādēts veiksmīgi!`);
    } else if (status === "error") {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
  onDrop(e) {
    console.log("Dropped files", e.dataTransfer.files);
  },
};


const ResidenceObjectForm = ({ objectId, groupId, residence }: { objectId: string, groupId: string, residence: any }) => {

  const router = useRouter();

  const { data: session, status, update } = useSession();

  const [form] = Form.useForm();

  console.log("residence", residence);

  form.setFieldsValue(residence);

  console.log(residence?.pictures)
  interface ObjectData {
    name: string;
    id: string;
    description: string;
    pictures: string[];
    groupId: string;
    createdAt: Date;
    updatedAt: Date;
    address: string;
  }

  const handleSubmit = async (values: any) => {
    console.log("Received values of form: ", values);

    const submitFunction = objectId !== "new" ? updateObjectApi : createObjectApi;

    const picturePromises = values.pictures.map(async (file: any) => {

        console.log("FILE", file);
      if (file.status === 'done') {
            const compressedFile = await imageCompression(file.originFileObj, {
                maxSizeMB: 0.05,
                maxWidthOrHeight: 400,
            });

            try {
                // Step 1: Get Pre-signed PUT URL
                const uploadUrlResults = await generateUploadUrl(compressedFile.name, "object-pictures");

                if (typeof uploadUrlResults !== "object" || "error" in uploadUrlResults) {
                    throw new Error("Failed to get upload URL");
                }

                const { presignedUrl, objectKey } = uploadUrlResults;

                if (typeof presignedUrl !== "string" || !objectKey) {
                    throw new Error("Failed to get upload URL");
                }

                console.log("Pre-signed Upload URL:", presignedUrl);

                // Step 2: Upload Image Using Pre-signed URL
                const uploadResponse = await fetch(presignedUrl, {
                    method: "PUT",
                    body: compressedFile, // Send the raw file
                });

                if (!uploadResponse.ok) {
                    throw new Error("Failed to upload image to MinIO");
                }

                console.log("Uploaded Image Key:", objectKey);

                // Save objectKey for further processing
                return { url: objectKey, status: 'new' };
            } catch (error) {
                console.error("Error uploading image to MinIO:", error);
                message.error("Image upload failed");
                throw error; // Propagate error to halt processing
            }
        } else if (file.status === 'deleted') {
            return { url: file.originFileObj, status: 'deleted' };
        } else {
            return { url: file.originFileObj, status: 'unchanged' };
        }
    });

    try {
        const pictures = await Promise.all(picturePromises);
        console.log("Uploaded Pictures:", pictures);

        const res = await submitFunction(groupId as string, {
            ...values,
            pictures,
        });

        console.log("res", res);

        if ("error" in res) {
            message.error("Failed to create object");
        } else {
            router.push(`/groups/${groupId}`);
        }
    } catch (error) {
        console.error("Error processing images:", error);
    }
};


  const normFile = (e: any) => {
    if (Array.isArray(e)) {
      return e;
    }
    return e && e.fileList;
  };


  
  if (status === 'loading') {
    return <div>Loading...</div>;
  }


  return (
    <Form
      form={form}
      layout="horizontal"
      onFinish={handleSubmit}
      className={styles["form-container"]}
    >
      <Row gutter={[84, 64]}
      >
      {/* Left Side */}
      <Col span={12}>
      {/* Name */}
      <Form.Item
        name="name"
        // label={<span className={styles["label"]}>Name</span>}
        rules={[
          {
            required: true,
            message: "Please input the name of the real estate object!",
          },
        ]}
        className={styles["form-item"]}
      >
        {/* <Input placeholder="Enter the name" className={styles["input-field"]} /> */}
        <StyledTextField 
          style={{ width: "100%" }}
          id="filled-basic-name"
          label="Objekta nosaukums"
          variant="outlined"
          defaultValue={residence.name}
          placeholder="Ievadiet objekta nosaukumu"
        />
      </Form.Item>

      {/* Description */}
      <Form.Item
        name="description"
        // label={<span className={styles["label"]}>Description</span>}
        rules={[
          {
            required: true,
            message: "Please input the description of the real estate object!",
          },
        ]}
      >
        <StyledTextField
          style={{ width: "100%" }}
          id="outlined-basic"
          label="Apraksts"
          variant="outlined"
          defaultValue={residence.description}
          placeholder="Ievadiet objekta aprakstu"
          multiline
          rows={3}
          // error={form.getFieldError('description').length > 0}
          // error={true}
          error={!!form.getFieldError("name").length} // Checks if there are errors
          helperText={form.getFieldError('description')}
          // helperText="Aprakstam jābūt vismaz 10 simbolus garumā"
        />
      </Form.Item>

      {/* Address */}
      <Form.Item
        name="address"
        // label={<span className={styles["label"]}>Address</span>}
        rules={[{ required: true, message: "Please input the address!" }]}
      >
        <StyledTextField
          style={{ width: "100%" }}
          id="outlined-basic"
          label="Adrese"
          variant="outlined"
          defaultValue={residence.address}
          placeholder="Ievadiet objekta adresi"
        />
      </Form.Item>

      {/* Area */}
      <Form.Item
        name="area"
        rules={[{ required: true, message: "Please input the area!" }]}
      >
        <StyledNumberInput
          style={{ width: "100%" }}
          id="outlined-basic"
          label="Platība (kvadrātmetri)"
          variant="outlined"
          defaultValue={residence.area}
          placeholder="Ievadiet objekta platību"
        />
      </Form.Item>

        <Row gutter={16}>
          <Col span={8}>
      {/* Bedroom Count */}
      <Form.Item
        name="bedroomCount"
        rules={[
          { required: true, message: "Please input the number of bedrooms!" },
        ]}
      >
        <StyledNumberInput
          style={{ width: "100%" }}
          id="outlined-basic"
          label="Guļamistabu skaits"
          variant="outlined"
          defaultValue={residence.bedroomCount}
          placeholder="Ievadiet guļamistabu skaitu"
        />
      </Form.Item>
      </Col>

      {/* Bathroom Count */}
      <Col span={8}>
      <Form.Item
        name="bathroomCount"
        rules={[
          {
            required: true,
            message: "Please input the number of bathrooms!",
          },
        ]}
      >
        <StyledNumberInput
          style={{ width: "100%" }}
          id="outlined-basic"
          label="Vannas istabu skaits"
          variant="outlined"
          defaultValue={residence.bathroomCount}
          placeholder="Ievadiet vannas istabu skaitu"
        />
      </Form.Item>
      </Col>

      {/* Parking Count */}
      <Col span={8}>
      <Form.Item
        name="parkingCount"
        rules={[
          {
            required: true,
            message: "Please input the number of parking spaces!",
          },
        ]}
      >
        <label style={{ color: "var(--background-light-main)", fontWeight: 'bold' }}>
        Stāvvietas pieejamas
      </label>
        <StyledSwitch/>
      </Form.Item>
      </Col>
      </Row>
      {/* Price */}
      <Form.Item
        name="price"
        rules={[{ required: true, message: "Please input the price!" }]}
      >
        <StyledNumberInput
          style={{ width: "100%" }}
          id="outlined-basic"
          label="Cena (EUR)"
          variant="outlined"
          defaultValue={residence.price}
          placeholder="Ievadiet objekta cenu"
        />
      </Form.Item>

      </Col>
      {/* Right Side */}
      <Col span={12}>
      {/* Pictures */}
      <Form.Item
        name="pictures"
        // label={<span className={styles["label"]}>Pictures</span>}
        valuePropName="fileList"
        getValueFromEvent={normFile}
        //   extra="Upload pictures of the real estate object"
      >
        <Dragger {...props}>
          <p className={styles["upload-drag-icon"]}>
            <InboxOutlined />
          </p>
          <p className={styles["upload-text"]}>
            Click or drag file to this area to upload
          </p>
          <p className={styles["upload-hint"]}>
            Support for a single or bulk upload. Strictly prohibited from
            uploading company data or other banned files.
          </p>
        </Dragger>
      </Form.Item>
      </Col>
      </Row>
      {/* Buttons */}
      <div className={styles["button-container"]}>
        <Button
          type="default"
          onClick={() => router.push(`/groups/${groupId}`)}
          className={styles["cancel-button"]}
        >
          Atcelt
        </Button>

        <Button
          type="primary"
          htmlType="submit"
          className={styles["submit-button"]}
        >
          Izveidot objektu
        </Button>
      </div>
    </Form>
  );
};

export default ResidenceObjectForm;
