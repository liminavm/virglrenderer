// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Load all three proc tables against whatever ICD the environment selects and report what each
//! answered.
//!
//! Not a gate -- a gate needs a driver, and this needs to run before there is a handler that
//! could stand one up. It is how the generated signatures get called for real before three
//! hundred handlers are built on top of them: an instance and a device created through the table
//! is the whole ABI claim, tested once.
//!
//! `VK_ICD_FILENAMES=<icd.json> cargo run --release --example probe`

use virglrenderer::venus::proto::types::*;
use virglrenderer::vulkan;

fn main() {
    let g = vulkan::global();
    println!("global   {:?}", g.loaded());

    let mut version: u32 = 0;
    // SAFETY: the loader answered for this name, and `version` is a live u32.
    let r = unsafe { (g.vkEnumerateInstanceVersion())(&mut version) };
    println!(
        "instance version {}.{}.{} (r={})",
        version >> 22,
        (version >> 12) & 0x3ff,
        version & 0xfff,
        r.0
    );

    let app = VkApplicationInfo { apiVersion: 1 << 22 | 3 << 12, ..Default::default() };
    let ci = VkInstanceCreateInfo { pApplicationInfo: &app, ..Default::default() };
    let mut inst = VkInstance(0);
    // SAFETY: `ci` and `inst` outlive the call; a null allocator is what the spec means by
    // "use the default".
    let r = unsafe { (g.vkCreateInstance())(&ci, core::ptr::null(), &mut inst) };
    assert_eq!(r.0, 0, "vkCreateInstance");
    let i = vulkan::instance(inst);
    println!("instance {:?}", i.loaded());

    let mut n: u32 = 0;
    // SAFETY: the count query with a null array is the spec's own two-call idiom.
    unsafe { (i.vkEnumeratePhysicalDevices())(inst, &mut n, core::ptr::null_mut()) };
    let mut pds = vec![VkPhysicalDevice(0); n as usize];
    // SAFETY: `pds` has room for `n`, which is what the count query just said.
    unsafe { (i.vkEnumeratePhysicalDevices())(inst, &mut n, pds.as_mut_ptr()) };
    println!("physical devices {n}");

    let pd = pds[0];
    let mut props = VkPhysicalDeviceProperties::default();
    // SAFETY: `props` is a live, correctly sized struct.
    unsafe { (i.vkGetPhysicalDeviceProperties())(pd, &mut props) };
    let name = props.deviceName.iter().take_while(|c| **c != 0).map(|c| *c as u8 as char);
    println!("device 0: {}", name.collect::<String>());

    let prio = 1.0f32;
    let q = VkDeviceQueueCreateInfo {
        sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueCount: 1,
        pQueuePriorities: &prio,
        ..Default::default()
    };
    let dci = VkDeviceCreateInfo {
        sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount: 1,
        pQueueCreateInfos: &q,
        ..Default::default()
    };
    let mut dev = VkDevice(0);
    // SAFETY: `dci` and everything it points at outlive the call.
    let r = unsafe { (i.vkCreateDevice())(pd, &dci, core::ptr::null(), &mut dev) };
    assert_eq!(r.0, 0, "vkCreateDevice");
    let d = vulkan::device(&i, dev);
    println!("device   {:?}", d.loaded());

    // SAFETY: handles this process created, destroyed in Vulkan's required order.
    unsafe {
        (d.vkDestroyDevice())(dev, core::ptr::null());
        (i.vkDestroyInstance())(inst, core::ptr::null());
    }
    println!("ok");
}
