/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.giraph.examples;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.giraph.conf.ImmutableClassesGiraphConfigurable;
import org.apache.giraph.conf.ImmutableClassesGiraphConfiguration;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.formats.TextVertexInputFormat;
import org.apache.giraph.utils.ArrayListWritable;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.log4j.Logger;

import com.google.common.collect.Lists;

/**
 * An algorithm which finds triangles in the graph
 * <p>
 * This is a fairly simple algorithm which finds triangles of three distinct
 * vertices in the graph.
 * </p>
 */
public class TriangleFindingComputation
  extends BasicComputation<LongWritable,
  TriangleFindingComputation.LongArrayArrayListWritable,
  NullWritable, TriangleFindingComputation.LongArrayWritable> {

  /**
   * Logger
   */
  private static final Logger LOG = Logger
      .getLogger(TriangleFindingComputation.class);

  @Override
  public void compute(Vertex<LongWritable, LongArrayArrayListWritable,
          NullWritable> vertex,
      Iterable<LongArrayWritable> messages) throws IOException {

    if (getSuperstep() == 0) {
      vertex.setValue(new LongArrayArrayListWritable());

      // Send initial message to all reachable vertices
      if (LOG.isDebugEnabled()) {
        LOG.debug("Sending initial message from vertex " + vertex.getId());
      }
      this.appendAndSend(vertex, new LongWritable[0]);
    } else {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Processing messages at vertex " + vertex.getId() +
              " in superstep " + getSuperstep());
      }

      Iterator<LongArrayWritable> iter = messages.iterator();
      while (iter.hasNext()) {
        LongArrayWritable msg = iter.next();
        LongWritable[] data = msg.getLongs();

        // Discard messages which contain duplicates
        if (this.hasDuplicate(data)) {
          if (LOG.isTraceEnabled()) {
            LOG.trace("Discarded message " + toString(data) +
                  " which contained duplicate vertices");
          }
          continue;
        }

        if (LOG.isTraceEnabled()) {
          LOG.trace("Processing message " + toString(data) +
                " at vertex " + vertex.getId());
        }

        switch (data.length) {
        case 1:
          // Add our ID to the array and send it out as a message
          this.appendAndSend(vertex, data);
          break;
        case 2:
          // Is there a triangle, we can detect this if any edge
          // corresponds to the first ID in the message
          Iterator<Edge<LongWritable, NullWritable>> edgeIter
            = vertex.getEdges().iterator();
          while (edgeIter.hasNext()) {
            Edge<LongWritable, NullWritable> e = edgeIter.next();
            if (e.getTargetVertexId().get() == data[0].get()) {
              this.appendAndSend(vertex, data, e);
            }
          }
          break;
        case 3:
          // Is this an actual triangle ?
          // i.e. is the first vertex ID the same as ours?
          if (data[0].get() == vertex.getId().get()) {
            if (!alreadyFound(vertex, data)) {
              if (LOG.isTraceEnabled()) {
                LOG.trace("Found triangle " + msg.toString());
              }
              // NB - Must create a new instance here as Giraph
              // will reuse
              // the LongArrayWritable to read the next message so
              // we'll
              // lose the discovered answers if we add the
              // original message
              // reference
              vertex.getValue().add(new LongArrayWritable(data));
            }
          }
          break;
        default:
          // Any other size message is invalid and discarded
          if (LOG.isTraceEnabled()) {
            LOG.trace("Invalid message " + toString(data));
          }
        }
      }
    }

    vertex.voteToHalt();
  }

  /**
   * Determines whether a triangle has already been found
   *
   * @param vertex
   *      Vertex
   * @param data
   *      Candidate triangle data
   * @return True if the given triangle has already been found, false
   *     otherwise
   */
  private boolean alreadyFound(Vertex<LongWritable, LongArrayArrayListWritable,
          NullWritable> vertex, LongWritable[] data) {
    LongArrayArrayListWritable value = vertex.getValue();

    if (LOG.isTraceEnabled()) {
      LOG.trace("Checking if triangle " + toString(data) +
            " is already known at vertex " + vertex.getId());
    }

    if (value.size() == 0) {
      return false;
    }
    for (int i = 0; i < value.size(); i++) {
      LongWritable[] existing = value.get(i).getLongs();
      int equals = 0;
      for (int j = 0; j < existing.length; j++) {
        if (existing[j].get() == data[j].get()) {
          equals++;
        }
      }
      if (equals == 3) {
        if (LOG.isTraceEnabled()) {
          LOG.trace("Triangle " + toString(data) + " equals triangle " +
            toString(existing) + " already known at vertex " +
              vertex.getId());
        }
        return true;
      }
    }
    return false;
  }

  /**
   * Gets the string form of the array for logging
   *
   * @param data
   *      Data
   * @return String form
   */
  private String toString(LongWritable[] data) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < data.length; i++) {
      builder.append(Long.toString(data[i].get()));
      if (i < data.length - 1) {
        builder.append(",");
      }
    }
    return builder.toString();
  }

  /**
   * Checks that the array contains no duplicates
   *
   * @param data
   *      Data
   * @return True if any duplicates, false otherwise
   */
  private boolean hasDuplicate(LongWritable[] data) {
    if (data.length <= 1) {
      return false;
    }
    Set<Long> values = new HashSet<Long>();
    for (int i = 0; i < data.length; i++) {
      if (values.contains(data[i].get())) {
        return true;
      }
      values.add(data[i].get());
    }
    return false;
  }

  /**
   * Appends the current vertex ID to the existing message data and sends it
   * out as a new message
   *
   * @param vertex
   *      Vertex ID
   * @param data
   *      Data
   */
  private void appendAndSend(Vertex<LongWritable, LongArrayArrayListWritable,
          NullWritable> vertex, LongWritable[] data) {
    LongWritable[] newData = Arrays.copyOf(data, data.length + 1);
    newData[newData.length - 1] = vertex.getId();

    if (this.hasDuplicate(newData)) {
      return;
    }

    Iterator<Edge<LongWritable, NullWritable>> iter
      = vertex.getEdges().iterator();
    while (iter.hasNext()) {
      Edge<LongWritable, NullWritable> e = iter.next();
      this.send(vertex, newData, e);
    }
  }

  /**
   * Appends the current vertex ID to the given message
   * and send it to target vertex of the given edge
   * @param vertex Vertex
   * @param data Existing message
   * @param e Edge
   */
  private void appendAndSend(Vertex<LongWritable, LongArrayArrayListWritable,
          NullWritable> vertex, LongWritable[] data,
          Edge<LongWritable, NullWritable> e) {
    LongWritable[] newData = Arrays.copyOf(data, data.length + 1);
    newData[newData.length - 1] = vertex.getId();

    if (this.hasDuplicate(newData)) {
      return;
    }
    this.send(vertex, newData, e);
  }

  /**
   * Sends the given message to the target vertex of
   * the given edge
   * @param vertex Vertex
   * @param data Message
   * @param e Edge
   */
  private void send(Vertex<LongWritable, LongArrayArrayListWritable,
          NullWritable> vertex, LongWritable[] data,
          Edge<LongWritable, NullWritable> e) {
    LongArrayWritable msg = new LongArrayWritable(data);
    if (LOG.isTraceEnabled()) {
      LOG.trace("Sending message " + msg + " from vertex " +
        vertex.getId() + " to vertex " +
        Long.toString(e.getTargetVertexId().get()));
    }
    this.sendMessage(e.getTargetVertexId(), msg);
  }

  /**
   * A writable array of {@link LongWritable}
   */
  public static class LongArrayWritable extends ArrayWritable {

    /**
     * Creates a new instance
     */
    public LongArrayWritable() {
      super(LongWritable.class);
    }

    /**
     * Creates a new instance
     *
     * @param data
     *      Data
     */
    public LongArrayWritable(LongWritable[] data) {
      super(LongWritable.class, data);
    }

    /**
     * Gets the array data as an array of {@link LongWritable}
     *
     * @return Data
     */
    public LongWritable[] getLongs() {
      Writable[] rawData = this.get();
      LongWritable[] data = new LongWritable[rawData.length];
      for (int i = 0; i < data.length; i++) {
        data[i] = (LongWritable) rawData[i];
      }
      return data;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      LongWritable[] ls = this.getLongs();
      for (int i = 0; i < ls.length; i++) {
        builder.append(Long.toString(ls[i].get()));
        if (i < ls.length - 1) {
          builder.append(',');
        }
      }
      return builder.toString();
    }
  }

  /**
   * A writable list of {@link LongArrayWritable}
   *
   */
  @SuppressWarnings("serial")
  public static class LongArrayArrayListWritable
    extends ArrayListWritable<LongArrayWritable> {

    @Override
    public void setClass() {
      super.setClass(LongArrayWritable.class);
    }
  }

  /**
   * Input format for triangle finding computation
   *
   */
  public static class TriangleFindingInputFormat extends
      TextVertexInputFormat<LongWritable, LongArrayArrayListWritable,
      NullWritable> implements
      ImmutableClassesGiraphConfigurable<LongWritable,
      LongArrayArrayListWritable, NullWritable> {

    /** Configuration. */
    private ImmutableClassesGiraphConfiguration<LongWritable,
      LongArrayArrayListWritable, NullWritable> conf;

    @Override
    public void setConf(
        ImmutableClassesGiraphConfiguration<LongWritable,
        LongArrayArrayListWritable, NullWritable> configuration) {
      this.conf = configuration;
    }

    @Override
    public ImmutableClassesGiraphConfiguration<LongWritable,
      LongArrayArrayListWritable, NullWritable> getConf() {
      return conf;
    }

    @Override
    public TextVertexReader createVertexReader(InputSplit split,
            TaskAttemptContext context) throws IOException {
      return new TriangleFindingInputReader();
    }

    /**
     * Input reader for triangle finding computation
     *
     */
    public class TriangleFindingInputReader extends
        TextVertexInputFormat<LongWritable, LongArrayArrayListWritable,
        NullWritable>.TextVertexReader {

      /** Separator of the vertex and neighbors */
      private final Pattern separator = Pattern.compile("[\t ]");

      @Override
      public boolean nextVertex() throws IOException, InterruptedException {
        return getRecordReader().nextKeyValue();
      }

      @Override
      public Vertex<LongWritable, LongArrayArrayListWritable,
        NullWritable> getCurrentVertex() throws IOException,
          InterruptedException {
        Vertex<LongWritable, LongArrayArrayListWritable, NullWritable>
          vertex = conf.createVertex();

        String[] tokens = separator.split(getRecordReader()
                .getCurrentValue().toString());
        List<Edge<LongWritable, NullWritable>> edges
          = Lists.newArrayListWithCapacity(tokens.length - 1);
        for (int n = 1; n < tokens.length; n++) {
          edges.add(EdgeFactory.create(new LongWritable(Long
                  .parseLong(tokens[n])), NullWritable.get()));
        }

        LongWritable vertexId = new LongWritable(Long.parseLong(tokens[0]));
        vertex.initialize(vertexId, new LongArrayArrayListWritable(), edges);

        return vertex;
      }
    }
  }
}
