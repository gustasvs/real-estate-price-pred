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
} from "antd";
import { InboxOutlined, UploadOutlined } from "@ant-design/icons";

import imageCompression from "browser-image-compression";

import styles from "./ResidenceObjectForm.module.css";
import {
  createObject as createObjectApi,
  getObject as getObjectApi,
  updateObject as updateObjectApi,
} from "../../../actions/groupObjects";

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
      message.success(`${info.file.name} file uploaded successfully.`);
    } else if (status === "error") {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
  onDrop(e) {
    console.log("Dropped files", e.dataTransfer.files);
  },
};

const ResidenceObjectForm = () => {
  const router = useRouter();
  const params = useParams();

  const group_id = Array.isArray(params.group_id)
    ? params.group_id[0]
    : params.group_id;
  const [form] = Form.useForm();
  interface ObjectData {
    name: string;
    id: string;
    description: string;
    pictures: string[];
    groupId: string;
    createdAt: Date;
    updatedAt: Date;
  }

  interface ErrorData {
    error: string;
  }

  const [initialData, setInitialData] = useState<ObjectData | ErrorData | null>(
    null
  );

  const objectId = params.object_id as string | undefined;

  useEffect(() => {
    // Function to fetch existing object data
    const fetchObjectData = async () => {
      if (objectId) {
        const data = await getObjectApi(objectId);
        if (!data) {
          return;
        }

        if ("pictures" in data) {
          form.setFieldsValue({
            ...data,
            pictures: data.pictures.map((url, index) => ({
              uid: index,
              name: `image-${index}.jpg`,
              status: "done",
              url, // existing image URL
            })),
          });
        }

        setInitialData(data);
      }
    };
    fetchObjectData();
  }, [form, objectId]);

  const handleSubmit = async (values: any) => {
    console.log("Received values of form: ", values);
    console.log("group_id", group_id);

    const submitFunction = objectId ? updateObjectApi : createObjectApi;

    const picturePromises = values.pictures.map(async (file: any) => {
      const compressedFile = await imageCompression(file.originFileObj, {
        maxSizeMB: 0.05,
        maxWidthOrHeight: 400,
      });

      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(compressedFile); // Convert compressed image to base64
        reader.onload = () => resolve(reader.result);
        reader.onerror = (error) => reject(error);
      });
    });
    const pictures = await Promise.all(picturePromises);

    const res = await submitFunction(group_id as string, {
      ...values,
      pictures,
    });

    console.log("res", res);

    if ("error" in res) {
      message.error("Failed to create object");
    } else {
      router.push(`/groups/${group_id}`);
    }
  };

  const normFile = (e: any) => {
    if (Array.isArray(e)) {
      return e;
    }
    return e && e.fileList;
  };

  return (
    <Form
      form={form}
      layout="horizontal"
      onFinish={handleSubmit}
      className={styles["form-container"]}
    >
      {/* Name */}
      <Form.Item
        name="name"
        label={<span className={styles["label"]}>Name</span>}
        rules={[
          {
            required: true,
            message: "Please input the name of the real estate object!",
          },
        ]}
        className={styles["form-item"]}
      >
        <Input placeholder="Enter the name" className={styles["input-field"]} />
      </Form.Item>

      {/* Address */}
      <Form.Item
        name="address"
        label={<span className={styles["label"]}>Address</span>}
        rules={[{ required: true, message: "Please input the address!" }]}
      >
        <Input
          placeholder="Enter the address"
          className={styles["input-field"]}
        />
      </Form.Item>

      {/* Area */}
      <Form.Item
        name="area"
        label={<span className={styles["label"]}>Area (in sq meters)</span>}
        rules={[{ required: true, message: "Please input the area!" }]}
      >
        <InputNumber
          min={0}
          placeholder="Enter the area"
          className={styles["input-field"]}
        />
      </Form.Item>

      {/* Bedroom Count */}
      <Form.Item
        name="bedroomCount"
        label={<span className={styles["label"]}>Bedrooms</span>}
        rules={[
          { required: true, message: "Please input the number of bedrooms!" },
        ]}
      >
        <InputNumber
          min={0}
          placeholder="Enter the number of bedrooms"
          className={styles["input-field"]}
        />
      </Form.Item>

      {/* Bathroom Count */}
      <Form.Item
        name="bathroomCount"
        label={<span className={styles["label"]}>Bathrooms</span>}
        rules={[
          {
            required: true,
            message: "Please input the number of bathrooms!",
          },
        ]}
      >
        <InputNumber
          min={0}
          placeholder="Enter the number of bathrooms"
          className={styles["input-field"]}
        />
      </Form.Item>

      {/* Parking Count */}
      <Form.Item
        name="parkingCount"
        label={<span className={styles["label"]}>Parking Spaces</span>}
        rules={[
          {
            required: true,
            message: "Please input the number of parking spaces!",
          },
        ]}
      >
        <InputNumber
          min={0}
          placeholder="Enter the number of parking spaces"
          className={styles["input-field"]}
        />
      </Form.Item>

      {/* Price */}
      <Form.Item
        name="price"
        label={<span className={styles["label"]}>Price</span>}
        rules={[{ required: true, message: "Please input the price!" }]}
      >
        <InputNumber
          min={0}
          placeholder="Enter the price"
          className={styles["input-field"]}
        />
      </Form.Item>

      {/* Predicted Price */}
      <Form.Item
        name="predictedPrice"
        label={<span className={styles["label"]}>Predicted Price</span>}
      >
        <InputNumber
          min={0}
          placeholder="Enter the predicted price"
          className={styles["input-field"]}
        />
      </Form.Item>

      {/* Description */}
      <Form.Item
        name="description"
        label={<span className={styles["label"]}>Description</span>}
        rules={[{ required: true, message: "Please input the description!" }]}
      >
        <Input.TextArea
          rows={4}
          placeholder="Enter the description"
          className={styles["input-field"]}
        />
      </Form.Item>

      <Form.Item
        name="pictures"
        label={<span className={styles["label"]}>Pictures</span>}
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
      <div className={styles["button-container"]}>
        <Button
          type="default"
          onClick={() => router.push(`/groups/${group_id}`)}
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
