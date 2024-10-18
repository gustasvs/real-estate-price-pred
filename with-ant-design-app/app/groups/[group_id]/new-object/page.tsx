"use client";

import React from "react";
import { useRouter, useParams } from "next/navigation";
import { Button, Form, Input, Upload, message, UploadProps } from "antd";
import { InboxOutlined, UploadOutlined } from "@ant-design/icons";
import GenericLayout from "../../../components/generic-page-layout";

import styles from "./NewObjectForm.module.css";
import { createObject as createObjectApi

} from "../../../../actions/groupObjects";

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

const NewObjectForm = () => {
  const router = useRouter();
  const params = useParams();

  const group_id = Array.isArray(params.group_id) ? params.group_id[0] : params.group_id;
  const [form] = Form.useForm();

  const handleSubmit = async (values: any) => {
    console.log("Received values of form: ", values);
    console.log("group_id", group_id);
  
    // Convert each image file to a base64 string
    const picturePromises = values.pictures.map((file: any) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file.originFileObj); // Convert to base64
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
      });
    });
  
    // Wait for all pictures to be converted
    const pictures = await Promise.all(picturePromises);
  
    // Send the data with the base64-encoded pictures
    const res = await createObjectApi(group_id as string, {
      ...values,
      pictures, // Replace file objects with base64 strings
    });

    console.log("res", res);

    if ('error' in res) {
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
    <GenericLayout>
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        // style={{ maxWidth: 600, margin: 'auto' }}
        className={styles["form-container"]}
      >
        <Form.Item
          name="name"
          label="Name"
          rules={[
            {
              required: true,
              message: "Please input the name of the real estate object!",
            },
          ]}
        >
          <Input placeholder="Enter the name" />
        </Form.Item>
        <Form.Item
          name="description"
          label="Description"
          rules={[{ required: true, message: "Please input the description!" }]}
        >
          <Input.TextArea rows={4} placeholder="Enter the description" />
        </Form.Item>
        <Form.Item
          name="pictures"
          label="Pictures"
          valuePropName="fileList"
          getValueFromEvent={normFile}
          extra="Upload pictures of the real estate object"
        >
          <Dragger {...props}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              Click or drag file to this area to upload
            </p>
            <p className="ant-upload-hint">
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
    </GenericLayout>
  );
};

export default NewObjectForm;
